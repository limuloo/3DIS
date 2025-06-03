import argparse
import logging
import traceback
import math
import os
os.environ["NCCL_P2P_DISABLE"] = "1"
import time
import random
from pathlib import Path
from typing import Iterable, Optional
import datetime
import numpy as np
import torch
print(f'[TRAIN DEBUG INFO] version of torch is {torch.__version__}')
import torch.nn.functional as F
import torch.utils.checkpoint
import sys
import cv2
from tensorboardX import SummaryWriter
import torch.nn as nn


sys.path.append('..')
if '/mnt/sda/zdw/project/MIGCtrain' in sys.path:
    sys.path.remove('/mnt/sda/zdw/project/MIGCtrain')
torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True
torch.set_printoptions(profile="full")
from ldm.models.diffusion.DDPMScheduler import DDPMScheduler
from diffusers import DDIMScheduler
from optimization import get_scheduler  # lr scheduler warmup
from itertools import repeat
from transformers import CLIPTokenizer, BertModel  # 换成CLIP的Tokenizer
# from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

from torchvision import transforms
from tqdm.auto import tqdm
from cfg_parser import parse_args, mox_copy

try:
    import moxing as mox
except ImportError:
    mox = None

is_torch_npu_available = False
 
def pyramid_noise_like(
    x, timesteps, strength=0.5, downscale_strategy="original", generator=None, device=None
):
    if torch.is_tensor(strength):
        strength = strength.reshape((-1, 1, 1, 1))
    b, c, w, h = x.shape

    if device is None:
        device = x.device

    up_sampler = torch.nn.Upsample(size=(w, h), mode="bilinear")
    noise = torch.randn(x.shape, device=x.device, generator=generator)

    # if "original" == downscale_strategy:
    for i in range(10):
        r = (
            torch.rand(1, generator=generator, device=device) * 2 + 2
        ) # Rather than always going 2x,
        w, h = max(1, int(w / (r**i))), max(1, int(h / (r ** i)))
        noise += (
            up_sampler(
                torch.randn(b, c, w, h, generator=generator, device=device).to(x)
            )
            * (timesteps[..., None, None, None] / 1000) * strength ** i
        )
        # print(timesteps.shape)
        if w == 1 or h == 1:
            break  # Lowest resolution is 1x1

    noise = noise / noise.std()  # Scaled back to roughly unit variance
    return noise


def multi_res_noise_like(
    x, strength=0.5, downscale_strategy="original", generator=None, device=None
):
    if torch.is_tensor(strength):
        strength = strength.reshape((-1, 1, 1, 1))
    b, c, w, h = x.shape

    if device is None:
        device = x.device

    up_sampler = torch.nn.Upsample(size=(w, h), mode="bilinear")
    noise = torch.randn(x.shape, device=x.device, generator=generator)

    if "original" == downscale_strategy:
        for i in range(10):
            r = (
                torch.rand(1, generator=generator, device=device) * 2 + 2
            )  # Rather than always going 2x,
            w, h = max(1, int(w / (r**i))), max(1, int(h / (r**i)))
            noise += (
                up_sampler(
                    torch.randn(b, c, w, h, generator=generator, device=device).to(x)
                )
                * strength**i
            )
            if w == 1 or h == 1:
                break  # Lowest resolution is 1x1
    elif "every_layer" == downscale_strategy:
        for i in range(int(math.log2(min(w, h)))):
            w, h = max(1, int(w / 2)), max(1, int(h / 2))
            noise += (
                up_sampler(
                    torch.randn(b, c, w, h, generator=generator, device=device).to(x)
                )
                * strength**i
            )
    elif "power_of_two" == downscale_strategy:
        for i in range(10):
            r = 2
            w, h = max(1, int(w / (r**i))), max(1, int(h / (r**i)))
            noise += (
                up_sampler(
                    torch.randn(b, c, w, h, generator=generator, device=device).to(x)
                )
                * strength**i
            )
            if w == 1 or h == 1:
                break  # Lowest resolution is 1x1
    elif "random_step" == downscale_strategy:
        for i in range(10):
            r = (
                torch.rand(1, generator=generator, device=device) * 2 + 2
            )  # Rather than always going 2x,
            w, h = max(1, int(w / (r))), max(1, int(h / (r)))
            noise += (
                up_sampler(
                    torch.randn(b, c, w, h, generator=generator, device=device).to(x)
                )
                * strength**i
            )
            if w == 1 or h == 1:
                break  # Lowest resolution is 1x1
    else:
        raise ValueError(f"unknown downscale strategy: {downscale_strategy}")

    noise = noise / noise.std()  # Scaled back to roughly unit variance
    return noise


def print_with_time(msg):
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("{}:INFO:{}".format(t, msg))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Adapted from torch-ema https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py#L14
class EMAModel(nn.Module):
    """
    Exponential Moving Average of models weights
    """

    def __init__(self, parameters: Iterable[torch.nn.Parameter], decay=0.99):
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]

        self.decay = decay
        self.optimization_step = 0

    def __getitem__(self, i):
        return self.shadow_params[i]
    
    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        value = (1 + optimization_step) / (10 + optimization_step)
        return 1 - min(self.decay, value)

    @torch.no_grad()
    def step(self, parameters):
        parameters = list(parameters)

        self.optimization_step += 1
        self.decay = self.get_decay(self.optimization_step)

        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                tmp = self.decay * (s_param - param)
                s_param.sub_(tmp)
            else:
                tmp = self.decay * (s_param - param)
                s_param.sub_(tmp)

        torch.cuda.empty_cache()

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Copy current averaged parameters into given collection of parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.

        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        self.shadow_params = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.shadow_params
        ]


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda")
    return torch.device("cpu")

def get_model_value(model, key):
    key_list = key.split('.')
    for key in key_list:
        if key.isdigit():
            model = model[int(key)]
        else:
            model = getattr(model, key)
    return model
from diffusers.models.attention_processor import Attention, AttnProcessor
if is_torch_npu_available:
    from diffusers.models.attention_processor import AttnProcessorNPU
if not is_torch_npu_available:
    from diffusers.models.attention_processor import XFormersAttnProcessor
from diffusers import AutoencoderKL
from migc.migc_arch import MIGC
MIGC_book = {'migc': MIGC}
class AttnProcessor_MIGC_TRAIN(nn.Module):
    r"""
    Default processor for performing attention-related computations.
    """
    def __init__(self, not_use_migc=True, C=1280, context_dim=1280, migc_type='migc', place=None):
        super().__init__()
        if is_torch_npu_available:
            self.ori_processor = AttnProcessorNPU()
        else:
            self.ori_processor = XFormersAttnProcessor()
        # self.ori_processor = XFormersAttnProcessor()
        self.not_use_migc = not_use_migc
        if not not_use_migc:
            self.migc = MIGC_book[migc_type](C, context_dim=context_dim)
            self.place = place

    def __call__(
            self,
            attn: Attention,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            scale=1.0,
            ith=None,
            embeds_pooler=None,
            timestep=None,
            height=512,
            width=512,
            box = None,
            guidance_mask=None,
            dilated_guidance_mask=None,
            MIG_text_tokens=None,
            supplement_mask=None,
            fuser_info_list=None,
            pre_attn_loss_MIGC_Place=None,
            # la_no_mask=False
    ):

        if self.not_use_migc:
            return self.ori_processor(attn, hidden_states, encoder_hidden_states, attention_mask, temb, scale)
        else:
            batch_size, instance_num, sequence_length, C = MIG_text_tokens.shape
            encoder_hidden_states = MIG_text_tokens.view(batch_size * instance_num, sequence_length, C)
            batch_size, HW, img_C = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, 1, HW, img_C).repeat(1, instance_num, 1, 1)
            ori_img_token = hidden_states
            hidden_states = hidden_states.view(batch_size * instance_num, HW, img_C)


            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)

            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            if pre_attn_loss_MIGC_Place is not None and self.place in pre_attn_loss_MIGC_Place:
                attention_probs_view = attention_probs.view(batch_size * instance_num, attn.heads, HW, 77)
                fuser_info_list.append({'pre_attn': attention_probs_view})
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)


            hidden_states = hidden_states / attn.rescale_output_factor
            hidden_states = hidden_states.view(batch_size, instance_num, HW, img_C)
            # assert encoder_hidden_states[0, 0, 0] == encoder_hidden_states[0, 1, 0]
            hidden_states = self.migc(
                ca_x = hidden_states,
                guidance_mask = guidance_mask[:, 1:, ...],
                other_info = {
                    'height': height,
                    'width': width,
                    'supplement_mask': supplement_mask,
                    'image_token': ori_img_token,
                    'sot': encoder_hidden_states[0, 0, :],
                    'context_pooler': embeds_pooler,
                    'box': box[:, 1:, :],
                    # 'la_no_mask': la_no_mask,
                    'dilated_guidance_mask': dilated_guidance_mask[:, 1:, ...] if dilated_guidance_mask is not None else None

                }
            )[:, 0, ...]


        return hidden_states

def load_model(device, args): #, config):
    # OmegaConf.update(config, "unet_config.params.use_fp16", args.use_fp16)
    # OmegaConf.update(config, "unet_config.params.input_act_ckpt", args.input_act_ckpt)
    # OmegaConf.update(config, "unet_config.params.output_act_ckpt", args.output_act_ckpt)
    # OmegaConf.update(config, "cond_stage_config.params.path", args.tokenizer_path)
    # if dist.get_rank() == 0:
    #     print(config)
    from diffusers import UNet2DConditionModel
    args.depth_migc_base_model = '/mnt/sda/zdw/ckpt/ldm3d-4c' if os.path.isdir('/mnt/sda/zdw/ckpt/ldm3d-4c') else 'Intel/ldm3d-4c'
    print(f'base model: {args.depth_migc_base_model}.')
    unet = UNet2DConditionModel.from_pretrained(
        args.depth_migc_base_model, subfolder="unet", revision=None
    )
    print('success load base model')
    print(f'load fine-tune unet {args.depth_state_dict}')
    if args.depth_state_dict is not None:
        unet.load_state_dict(torch.load(args.depth_state_dict, map_location='cpu'))
    print('success load fine-tune model')
    def get_sd_xl_attn_key(unet):
        all_sd_xl_attn_key = []
        for key in unet.state_dict().keys():
            if 'attn1' in key:
                key = key.split('.attn1.')[0] + '.attn1.processor'
                all_sd_xl_attn_key.append(key)
            if 'attn2' in key:
                key = key.split('.attn2.')[0] + '.attn2.processor'
                all_sd_xl_attn_key.append(key)
        all_sd_xl_attn_key = list(set(all_sd_xl_attn_key))
        all_sd_xl_attn_key = sorted(all_sd_xl_attn_key)
        return all_sd_xl_attn_key

    sd_xl_attn_key = get_sd_xl_attn_key(unet)
    processor_dict = {}
    for key in sd_xl_attn_key:
        processor_dict[key] = AttnProcessor_MIGC_TRAIN(migc_type=args.migc_type)
    if args.need_MIGC_pretrained:
        MIGC_DICT = torch.load(args.MIGC_pretrained_path, map_location='cpu')
    for MIGC_place in args.MIGC_Place:
        if 'up_blocks.2' in MIGC_place or 'down_blocks.1' in MIGC_place:
            processor_dict[MIGC_place] = AttnProcessor_MIGC_TRAIN(C=640, not_use_migc=False, context_dim=768, migc_type=args.migc_type, place=MIGC_place)
        elif 'up_blocks.3' in MIGC_place or 'down_blocks.0' in MIGC_place:
            processor_dict[MIGC_place] = AttnProcessor_MIGC_TRAIN(C=320, not_use_migc=False, context_dim=768, migc_type=args.migc_type, place=MIGC_place)
        else:
            processor_dict[MIGC_place] = AttnProcessor_MIGC_TRAIN(not_use_migc=False, context_dim=768, migc_type=args.migc_type, place=MIGC_place)
        if args.need_MIGC_pretrained:
            NOW_DICT = {}
            for key in MIGC_DICT.keys():
                key: str
                if key.startswith(MIGC_place) and (key.split(MIGC_place + '.')[-1].split('.')[1] in args.MIGC_preserve):
                    NOW_DICT[key.split(MIGC_place + '.')[-1]] = MIGC_DICT[key]
            processor_dict[MIGC_place].load_state_dict(NOW_DICT, strict=False)
            print('load', NOW_DICT.keys())
    print(processor_dict.keys())
    unet.set_attn_processor(processor_dict)
    unet.enable_gradient_checkpointing()
    print('put the unet to cuda')
    unet = unet.to(dev())
    print('finishing putting the unet to cuda')
    # unet = instantiate_from_config(config.unet_config)
    # unet.to(dev())

    if args.need_vae:
        ### Construct VAE ###
        ldm3d4c_path = '/mnt/sda/zdw/ckpt/ldm3d-4c' if os.path.isdir('/mnt/sda/zdw/ckpt/ldm3d-4c') else 'Intel/ldm3d-4c'
        vae = AutoencoderKL.from_pretrained(
            ldm3d4c_path,
            subfolder="vae"
        ).to(dev())

    if args.need_dino:
        from transformers import CLIPVisionModel
        dino_encoder = CLIPVisionModel.from_pretrained('../../pretrained_models/clip_tokenizer')
        dino_encoder = dino_encoder.to(dev())

    from transformers import CLIPTextModel, CLIPTextModelWithProjection
    sd14_path = '/mnt/sda/zdw/ckpt/new_sd14' if os.path.isdir('/mnt/sda/zdw/ckpt/new_sd14') else 'CompVis/stable-diffusion-v1-4'
    # CompVis/stable-diffusion-v1-4
    text_encoder1 = CLIPTextModel.from_pretrained(
        sd14_path, subfolder="text_encoder"
    ).to(device)

    unet.requires_grad_(False)
    # 待选Freeze模块
    if args.need_vae:
        vae.requires_grad_(False)
    text_encoder1.requires_grad_(False)


    # 处理unet的参数，只选择adapter 参数改动，其余不动
    # TODO
    # unet.requires_grad_(True)
    for name, param in unet.named_parameters():
        name_list = name.split('.')
        if 'processor' in name_list:
            # if args.need_MIGC_pretrained:
            flag = False
            for train_part in args.MIGC_train:
                if train_part in name_list:
                    flag = True
                    break
            if flag:
                print(f'train: {name}')
                param.requires_grad_(True)


    if args.only_train_MuViC_partly:
        assert False
        for name, param in unet.named_parameters():
            ok_flag = False
            name_list = name
            for o in args.train_MuViC_partly_name:
                o = o.split('*')
                all_have = True
                for oo in o:
                    if oo not in name_list:
                        all_have = False
                if all_have:
                    ok_flag = True
            if not ok_flag:
                param.requires_grad_(False)
            else:
                print(name)

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    if args.use_zero:
        print_with_time('USING ZeRO')

        from fairscale import __version__ as __fs_version__
        from fairscale.optim import OSS

        oss_kwargs = {
            'broadcast_fp16': args.use_fp16
        }
        if __fs_version__ >= '0.4.6':
            oss_kwargs['force_broadcast_object'] = True

        optimizer = OSS(
            [
                {'params': text_encoder1.parameters()},
                {'params': unet.parameters()},
            ], optimizer_cls,
            lr=args.learning_rate, weight_decay=args.adam_weight_decay, betas=(args.adam_beta1, args.adam_beta2),
            **oss_kwargs
        )
        from fairscale.nn.data_parallel import ShardedDataParallel
        text_encoder1 = ShardedDataParallel(
            text_encoder1, optimizer, reduce_buffer_size=0,
            reduce_fp16=args.use_fp16
        )
        unet = ShardedDataParallel(
            unet, optimizer, reduce_buffer_size=0,
            reduce_fp16=args.use_fp16
        )
    else:
        # 设置非Freeze部分到opt
        optimizer = optimizer_cls(
            [
                {'params': text_encoder1.parameters()},
                {'params': unet.parameters()},
            ],
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        device_ids = [device]
        text_encoder1 = DDP(text_encoder1, device_ids=device_ids)
        unet = DDP(unet, device_ids=device_ids)
    ret_data ={
        'text_encoder1': text_encoder1,
        'unet': unet,
        'optimizer': optimizer
    }
    if args.need_vae:
        ret_data['vae'] = vae
    if args.need_dino:
        ret_data['dino'] = dino_encoder
    return ret_data




def save(args, unet, epoch, step, ema_model=None):
    state = {}
    output_dir = os.path.join(args.output_dir, 'epoch' + str(epoch))
    output_path = os.path.join(output_dir, f"model{(step):06d}.ckpt")
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    state = {k.replace('module.', ''): v for k, v in unet.state_dict().items() if '.migc.' in k}
    torch.save(state, output_path)
    if args.use_ema:
        state_emas = state
        for i, name in enumerate(state):
            state_emas[name] = ema_model[i]
        torch.save(state_emas, output_path.replace('.ckpt', '_ema.ckpt'))


def init_process_group(args):
    torch_version = torch.__version__
    if not args.is_cloud:
        args.world_size = torch.cuda.device_count()

    try:
        print('zdw init_process_group:')
        print(
            f'[torch.distributed.launch INFO]: RANK:{os.environ["RANK"]}, LOCAL_RANK:{os.environ["LOCAL_RANK"]}, WORLD_SIZE:{os.environ["WORLD_SIZE"]}')
        print(
            f'[torch.distributed.launch INFO]: MASTER_ADDR:{os.environ["MASTER_ADDR"]}, MASTER_PORT:{os.environ["MASTER_PORT"]}')
    except Exception as e:
        print("zdw get os.environ error occurred: ", e)
        traceback.print_exc()

    if is_torch_npu_available:
        torch.distributed.init_process_group(backend='hccl',
                                            init_method='env://',
                                            timeout=datetime.timedelta(hours=0.1))
    else:
        torch.distributed.init_process_group(backend='nccl',
                                            init_method='env://',
                                            timeout=datetime.timedelta(hours=0.1))
    # torch.distributed.barrier()
    if args.unify_seed:
        print(f'set_seed:{args.global_seed + int(os.environ["RANK"])}')
        set_seed(args.global_seed + int(os.environ["RANK"]))

    try:
        print(f'[init_process_group INFO]: RANK:{dist.get_rank()}, WORLD_SIZE:{dist.get_world_size()}')
    except Exception as e:
        print("zdw init_process_group error occurred: ", e)
        traceback.print_exc()


def repeater(data_loader, seed_shift=0, sampler_set_epoch=1):
    for i, loader in enumerate(repeat(data_loader)):
        if sampler_set_epoch:
            loader.sampler.set_epoch(i + seed_shift)
        for data in loader:
            yield data


def main():
    args = parse_args()

    init_process_group(args)
    if dist.get_rank() == 0:  # 这个判断的是全局rank
        print(args)

    if dist.get_rank() == 0:  # 只需要全局rank第一的進程記錄loss即可
        # 根據配置文件的名稱決定實驗名
        exp_name = os.path.split(args.yml_path)[-1][:-5]
        # 获取当前时间
        now = datetime.datetime.now()
        # 添加时间信息
        exp_name = f"{exp_name}_{now.strftime('%Y%m%d_%H%M%S')}"
        # 保存log的folder路徑
        save_log_dir = os.path.join('../../zdw_logs/', exp_name)  # 記錄logs的位置, 暫時寫死'../zdw_logs/'
        print(f'[save log dir INFO] {save_log_dir}')
        if not os.path.isdir(save_log_dir):
            os.makedirs(save_log_dir)
        train_log_writer = SummaryWriter(save_log_dir)
    ngpus_per_node = torch.cuda.device_count()
    device = args.local_rank % ngpus_per_node
    torch.cuda.set_device(device)
    now_time = torch.from_numpy(np.array(int(time.time()))).cuda() / ngpus_per_node
    dist.all_reduce(now_time, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    from transformers import AutoTokenizer
    sd14_path = '../../../ckpt/new_sd14' if os.path.isdir('../../../ckpt/new_sd14') else 'CompVis/stable-diffusion-v1-4'
    # CompVis/stable-diffusion-v1-4
    tokenizer1 = AutoTokenizer.from_pretrained(
        sd14_path, subfolder="tokenizer", revision=None, use_fast=False
    )

    load_model_data = load_model(device, args) #, config)
    print('success load all model')
    text_encoder1 = load_model_data['text_encoder1']
    unet = load_model_data['unet']
    # {k.replace('module.', ''): v for k, v in unet.state_dict().items() if '.migc.' in k}
    if args.use_ema:
        ema_model = EMAModel({k: v for k, v in unet.state_dict().items() if '.migc.' in k}.values())
    
    optimizer = load_model_data['optimizer']

    if args.need_vae:
        vae = load_model_data['vae']

    if args.need_dino:
        dino = load_model_data['dino']

    if args.use_zero:
        from fairscale.optim.grad_scaler import ShardedGradScaler
        scaler = ShardedGradScaler(enabled=args.use_fp16)
    elif args.use_fp16:
        scaler = GradScaler(enabled=args.use_fp16)

    # 使用自定义DDPMScheduler
    noise_scheduler = DDPMScheduler(timesteps=1000, beta_schedule="linear", linear_start=0.00085, linear_end=0.012,
                                    cosine_s=8e-3)
    noise_scheduler.to(dev())

    from ldm.data.clip_dataloader_sam_v6_on_depth import CLIP_fea_zip_dataloader
    print('Init dataloader')
    train_dataloader = CLIP_fea_zip_dataloader(args, tokenizer1)
    print('Success Init dataloader')

    # 更换使用的数据集
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not dist.get_rank() == 0)
    progress_bar.set_description("Steps")
    global_step = 0

    if args.need_vae:
        vae.eval()
    unet.train()
    text_encoder1.eval()
    log_iter = 0
    log_loss = {"mse_loss": 0.0, 'cross_loss': 0.0, "bbox_loss_v1": 0.0}
    for epoch in range(args.num_train_epochs):
        print(f'epoch[{epoch}]:')
        for step, batch in enumerate(train_dataloader):
            # batch['pixel_values'] = F.interpolate(batch['pixel_values'], 64)
            # Convert images to latent space
            st = time.time()
            desc = batch['captions']
            # if dist.get_rank() == 0:
            #     print(desc)
            semantic_mask = batch['semantic_mask']
            # dilated_semantic_mask = batch['dilated_semantic_mask']
            if args.use_bbox_loss_v1:
                bbox_loss_v1_mask = torch.sum(semantic_mask[:, 1:, ...], dim=1, keepdim=True)
                bbox_loss_v1_mask[bbox_loss_v1_mask > 1] = 1
            # attn_mask = batch['attn_mask']
            '''
            attn_mask.shape: torch.Size([2, 512, 512, 10])
            2是batch size
            10是補全的token數量。mask_padding。
            '''
            # attn_label = batch['attn_token']
            # 找到每个mask 应该对应哪张map

            with torch.no_grad():
                if not args.use_feature_dataset:
                    assert args.need_vae, "Conflict!"
                    try:
                        latents = vae.encode(batch["pixel_values"]).sample()
                    except:
                        latents = vae.encode(batch["pixel_values"])
                else:
                    # if step == 0 and args.need_vae:
                    #     _ = vae.encode(torch.randn(args.train_batch_size, 3, 512, 512).to(vae.device)).sample()
                    latents = batch["pixel_values"].detach()
                latents = latents * args.latents_scale
                if args.use_bbox:
                    batch['boxes'] = F.interpolate(batch['boxes'], latents.shape[-2:])
            if args.use_zero or args.use_fp16:
                with autocast(enabled=args.use_fp16):
                    # Sample noise that we'll add to the latents


                    if latents.dim() >= 5:
                        latents = latents.view(latents.shape[0] * latents.shape[1], 4, 64, 64)
                    
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    if args.different_t_in_one_batch:
                        timesteps = torch.randint(args.time_st, args.time_ed, (bsz,), device=latents.device)
                        timesteps = timesteps.long()
                    else:
                        timesteps = torch.randint(args.time_st, args.time_ed, (1,), device=latents.device).repeat(bsz)
                        timesteps = timesteps.long()

                    if args.use_multi_res_noise:
                        noise = multi_res_noise_like(latents, strength=args.multi_res_noise_scale)
                    elif args.use_pyramid_noise:
                        # print('p')
                        noise = pyramid_noise_like(latents, timesteps, strength=args.pyramid_noise_scale)
                    else:
                        noise = torch.randn_like(latents)  # .to(weight_dtype)
                    
                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.q_sample(latents, timesteps, noise)  # .to(weight_dtype)
                    # Get the text embedding for conditioning
                    input_ids1 = batch["input_ids"]  # (B, PN, 77)

                    bsz, num_phase, _ = input_ids1.shape
                    input_ids1 = input_ids1.view(bsz * num_phase, _)
                    # encoder_hidden_states = text_encoder.module.encode_token(batch["input_ids"].cuda())
                    encoder_tokens1 = text_encoder1(
                        input_ids1.to(device),
                        output_hidden_states=True,
                    )
                    encoder_hidden_states = encoder_tokens1.last_hidden_state
                    encoder_pooler_states = encoder_tokens1.pooler_output
                    encoder_hidden_states = encoder_hidden_states.view(bsz, num_phase, _, 768)
                    encoder_pooler_states = encoder_pooler_states.view(bsz, num_phase, 1, 768)
                    if args.need_dino:
                        assert False
                        RGB_image = batch['RGB_image']
                        instance_img = batch['instance_img']  # (B, PN, C, H, W)
                        instance_img = instance_img.view(bsz * num_phase, 3, 224, 224)
                        instance_img = instance_img * 2 - 1
                        instance_img = dino(instance_img, output_hidden_states=True)
                        instance_img = instance_img[0].detach()
                        instance_img = instance_img.view(bsz, num_phase, 257, 1024)

                    if args.use_cross_uncond and np.random.uniform() < args.cross_uncond_p:
                        assert False
                        uncond_states = np.load("../../pretrained_models/sd/uncond_feature.npy")
                        uncond_states = [uncond_states.copy()[None, ...] for i in range(bsz)]
                        uncond_states = np.concatenate(uncond_states, axis=0)
                        uncond_states = torch.from_numpy(uncond_states).to(encoder_hidden_states.device)
                        encoder_hidden_states = torch.cat([encoder_hidden_states[..., None], uncond_states[..., None]], dim=-1)

                    # Get the target for loss depending on the prediction type `
                    target = noise
                    _, _, H, W = latents.shape
                    
                    
                    fuser_info_list = []
                    model_pred = unet(
                        noisy_latents,  # torch.Size([2, 4, 128, 128])
                        timesteps[0],  # tensor(981., device='cuda:0')
                        encoder_hidden_states=encoder_hidden_states[:, 0, ...],  # torch.Size([B, 77, 2048])
                        cross_attention_kwargs={
                            'ith': timesteps[0],
                            'embeds_pooler': encoder_pooler_states,  # torch.Size([B, instance_num, 1, 1280])
                            'timestep': timesteps[0],
                            'height': H * 8,
                            'width': W * 8,
                            'guidance_mask': semantic_mask,
                            'dilated_guidance_mask': batch['dilated_mask'] if args.dilate_mask else None,
                            'MIG_text_tokens': encoder_hidden_states,
                            'supplement_mask': batch["supplement_mask"],
                            'box': batch["box"],  # torch.Size([B, instance_num, 4])
                            'fuser_info_list': fuser_info_list,
                            'pre_attn_loss_MIGC_Place': args.pre_attn_loss_MIGC_Place
                            # 'la_no_mask': args.la_no_mask,
                        },  # None
                        # added_cond_kwargs={'text_embeds': encoder_pooler_states[:, 0, 0, :], 'time_ids': add_time_ids},
                        # {'text_embeds': torch.Size([B, 1280]), 'time_ids': torch.Size([B, 6])}
                        return_dict=False,
                    )[0]
                    # Calculate loss function
                    loss = 0.0
                    mse_loss = F.mse_loss(model_pred.float(), target.float(),
                                          reduction="mean") / args.gradient_accumulation_steps * \
                               args.mse_weight

                    # cross_loss = crossloss(attn_map, attn_mask) / args.gradient_accumulation_steps * args.cross_weight

                    log_loss["mse_loss"] += mse_loss.item()
                    # if cross_loss != 0.0:
                    #     log_loss['cross_loss'] += cross_loss.item()

                    # loss = loss + mse_loss + cross_loss
                    loss = loss + mse_loss

                    
                # Backpropagate
                scaler.scale(loss).backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if dist.get_rank() == 0:  # 全局rank0才需要輸出log
                        for loss_name in log_loss:
                            train_log_writer.add_scalar(loss_name, log_loss[loss_name], log_iter)
                            log_iter = log_iter + 1
                            log_loss[loss_name] = 0.0
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    lr_scheduler.step()
                    if args.use_ema:
                        ema_model.step({k: v for k, v in unet.state_dict().items() if '.migc.' in k}.values())
                    progress_bar.update(1)
                    global_step += 1
            else:
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)  # .to(weight_dtype)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.q_sample(latents, timesteps, noise)  # .to(weight_dtype)
                encoder_hidden_states = text_encoder.module.encode_token(batch["input_ids"].cuda())
                target = noise
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states.detach())
                loss = F.mse_loss(model_pred, target.detach(), reduction="mean") / args.gradient_accumulation_steps
                loss = loss * args.mse_weight
                # Backpropagate
                loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()
                    progress_bar.update(1)
                    global_step += 1
            et = time.time()
            cnt = batch["pixel_values"].size()[0] * torch.cuda.device_count()
            # logs = {"mse_loss": mse_loss, "cross_loss": cross_loss, "lr": lr_scheduler.get_last_lr()[0], 'FPS': cnt / (et - st)}
            logs = {"mse_loss": mse_loss, "lr": lr_scheduler.get_last_lr()[0],
                    'FPS': cnt / (et - st)}
            
            if args.use_pre_attn_loss:
                logs["pre_attn_loss"] = pre_attn_loss
            if dist.get_rank() == 0:
                progress_bar.set_postfix(**logs)

            if not args.is_cloud and step % 10 == 0 and step > 0 and dist.get_rank() == 0 and args.need_vae:
                with torch.no_grad():
                    print(desc)
                    choose_batch_id = np.random.randint(0, noisy_latents.shape[0])
                    print(f'choose {choose_batch_id}')
                    img = vae.decode((noisy_latents / args.latents_scale).float().detach())[0]
                    img = torch.clamp(img, -1, 1)
                    img = img.cpu().numpy()[choose_batch_id, ...].transpose((1, 2, 0))
                    img = (img + 1) / 2
                    img = img * 255.
                    img = img[..., :-1]
                    from PIL import Image
                    Image.fromarray(img.astype('uint8')).save(f'./debug_noise.png')

                    timesteps = timesteps[:, None, None, None]
                    x0 = noise_scheduler.get_x0(noisy_latents, model_pred, timesteps)
                    # x0 = torch.clamp(x0, -1, 1)
                    img = vae.decode((x0 / args.latents_scale).float().detach())[0]
                    img = torch.clamp(img, -1, 1)
                    img = img.cpu().numpy()[choose_batch_id, ...].transpose((1, 2, 0))
                    img = (img + 1) / 2
                    img = img * 255.
                    from PIL import Image
                    Image.fromarray(img.astype('uint8')).save(f'./debug_pred.png')

                    img = vae.decode((latents / args.latents_scale).float().detach())[0]
                    img = torch.clamp(img, -1, 1)
                    if args.use_bbox:
                        ######### 如果使用了bbox增强,在可视化的时候，把bbox位置也显示出来。 #########
                        print('original, mask.shape', batch['boxes'].shape)
                        upsample_mask = F.interpolate(batch['boxes'], img.shape[2:])
                        upsample_mask = upsample_mask.to(img.device)
                        print('upsample, mask.shape', upsample_mask.shape)
                        img = img * 0.1 + img * upsample_mask / upsample_mask.max() * 0.9
                    img = img.cpu().numpy()[choose_batch_id, ...].transpose((1, 2, 0))
                    img = (img + 1) / 2
                    img = img[..., :-1]

                    mask = F.interpolate(batch['dilated_mask'], size=(512, 512), mode=args.inter_mode)
                    
                    while True:
                        mask_idx = random.randint(1, args.phase_num - 1)
                        if desc[mask_idx][choose_batch_id] == '':
                            continue
                        else:
                            print(desc[mask_idx][choose_batch_id])
                            break
                    layout_box = batch["box"]
                    now_box = layout_box[choose_batch_id, mask_idx, ...].cpu().numpy() * 512
                    now_box = now_box.astype('uint16')
                    mask = mask[choose_batch_id, mask_idx, ...].cpu().numpy()[..., None]

                    img = img * 0.1 + img * mask * 0.9
                    img = (img * 255.).astype('uint8')
                    def my_draw_rectangle(img, W1, W2, H1, H2, width = 5):
                        for i in range(H1, H2):
                            for j in [W1, W2]:
                                for k in range(-width, width):
                                    ii = i + k
                                    jj = j + k
                                    if ii >= 0 and ii < img.shape[0] and jj >= 0 and jj < img.shape[1]:
                                        img[ii, jj, :] = (255, 0, 0)

                        for i in [H1, H2]:
                            for j in range(W1, W2):
                                for k in range(-width, width):
                                    ii = i + k
                                    jj = j + k
                                    if ii >= 0 and ii < img.shape[0] and jj >= 0 and jj < img.shape[1]:
                                        img[ii, jj, :] = (255, 0, 0)

                    my_draw_rectangle(img, now_box[0], now_box[1], now_box[2], now_box[3])
                    if args.use_bbox and args.use_bbox_caption_aug:
                        ######### 如果使用了use_bbox_caption_aug增强,在可视化的时候，把caption位置也显示出来。 #########
                        desc = desc[0]
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        cv2.putText(img, desc[: len(desc) // 4], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))
                        cv2.putText(img, desc[len(desc) // 4: len(desc) // 4 * 2], (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 255))
                        cv2.putText(img, desc[len(desc) // 4 * 2: len(desc) // 4 * 3], (50, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))
                        cv2.putText(img, desc[len(desc) // 4 * 3:], (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 0, 255))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    Image.fromarray(img.astype('uint8')).save(f'./debug_gt.png')

            if step == len(train_dataloader) - 10 and step > 0 and (epoch - 1) % args.save_epoch_freq == 0:
                if dist.get_rank() == 0:
                    save(args, unet, epoch, step, ema_model=ema_model if args.use_ema else None)

            if global_step >= args.max_train_steps:
                break
            

if __name__ == "__main__":
    main()