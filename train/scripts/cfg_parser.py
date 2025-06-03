import os
import re
import sys
import time
import yaml
import argparse
import subprocess
from pathlib import Path
import multiprocessing as mp

try:
    import moxing as mox
    mox.file.set_auth(is_secure=False)
except ImportError:
    mox = None


def list_from_str(str_val):
    str_val = str_val.strip(' ,[]()')
    list_val = filter(lambda x: len(x) > 0, str_val.split(','))
    list_val = list(map(int, list_val))
    return list_val

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    # ----------------------distributed parameter-----------------------
    parser.add_argument('--backend', type=str, default="nccl", help='use for current backend for distributed')
    parser.add_argument('--init_method', type=str, default="env://", help='init method for distributed')
    parser.add_argument('--local_rank', type=int, default=0, help='current rank for distributed')
    parser.add_argument('--world_size', type=int, default=8, help='current process number for distributed')

    parser.add_argument("--revision", type=str, default=None,
        required=False, help="Revision of pretrained model identifier from huggingface.co/models.",)
    parser.add_argument("--dataset_name", type=str, default=None,
        help=("The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."),)
    parser.add_argument("--dataset_config_name", type=str, default=None,
        help="The config of the Dataset, leave as None if there's only one config.",)
    parser.add_argument("--train_data_dir", type=str, default=None,
        help=("A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."),)
    parser.add_argument("--image_column", type=str, default="image", help="The column of the dataset containing an image.")
    parser.add_argument("--caption_column", type=str, default="text",
        help="The column of the dataset containing a caption or a list of captions.",)
    parser.add_argument("--max_train_samples", type=int, default=None,
        help=("For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."),)
    parser.add_argument("--output_dir", type=str, default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",)
    parser.add_argument("--cache_dir", type=str, default=None,
        help="The directory where the downloaded models and datasets will be stored.",)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")     # 暂未设置
    parser.add_argument("--resolution", type=int, default=512,
        help=("The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"),)
    parser.add_argument("--center_crop", action="store_true",
        help="Whether to center crop images before resizing to resolution (if not set, random crop will be used)",)
    parser.add_argument("--random_flip", action="store_true",
        help="whether to randomly flip images horizontally",)
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
        help="Number of updates steps to accumgradient_checkpointingulate before performing a backward/update pass.",)
    parser.add_argument("--learning_rate", type=float,
        default=1e-4, help="Initial learning rate (after the potential warmup period) to use.",)

    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help=('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'),)
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--use_ema", type=bool, default=False, help="Whether to use EMA model.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument("--hub_model_id", type=str, default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",)
    parser.add_argument("--logging_dir", type=str, default="logs",
        help=("[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."),)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],
        help=("Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."),)
    parser.add_argument("--report_to", type=str, default="tensorboard",
        help=('The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."),)
    parser.add_argument('--yml_path', type=str, default='', help='cfg name')
    parser.add_argument('--model_config', type=str, default='', help='model cfg name')
    parser.add_argument("--model_path", type=str, default='', help="Path to pretrained model")
    parser.add_argument("--tokenizer_path", type=str, default='/mnt/data1/dewei/ckpt/clip-vit-large-patch14', help="Path to tokenizer file")
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument('--local_shuffle_type', default=4, type=int, help="0: not use local shuffle "
                                                                          "1: use local shuffle by node "
                                                                          "2: use local shuffle by card "
                                                                          "4: use local shuffle by card in cloud")
    parser.add_argument('--zip_max_split', type=int, default=64, help="used when local_shuffle_type=4")
    # ZeRO
    parser.add_argument('--use_zero', default=False, type=bool, help='use Zero Redundancy Optimizer')
    # mixed precision training
    parser.add_argument('--use_fp16', default=False, type=bool)
    # activation checkpointing configuration

    parser.add_argument('--input_act_ckpt', default=None, type=str,
                        help="can be None, 'all' or str `i,j`, checkpoints U-Net input blocks from i to j")
    parser.add_argument('--output_act_ckpt', default=None, type=str,
                        help="can be None, 'all' or str `i,j`, checkpoints U-Net output blocks from i to j")
    parser.add_argument('--train_url', default='results/train_on_Wukong', type=str, help='dist folder')
    parser.add_argument('--save_interval', default=20000, type=int, )
    parser.add_argument('--profile', default=False, type=bool)
    parser.add_argument('--deborder', default=False, type=bool)
    parser.add_argument('--is_cloud', default=False, type=bool)
    parser.add_argument('--s3_result_folder', default='', type=str)
    parser.add_argument('--s3_data_root', default='', type=str)
    parser.add_argument('--idx_len', default=0, type=int)
    parser.add_argument('--latents_scale', default=1.0, type=float)
    parser.add_argument('--load_unet', default=False, type=bool)
    parser.add_argument('--mse_weight', default=1.0, type=float)
    parser.add_argument('--use_ce_loss', default=False, type=bool)
    parser.add_argument('--ce_weight', default=1.0, type=float)
    parser.add_argument('--only_upload_some_tar', default=False, type=bool)
    parser.add_argument('--upload_tar_num', default=100, type=int)
    parser.add_argument('--use_bbox', default=False, type=bool)
    parser.add_argument('--bbox_root', default='', type=str)
    parser.add_argument('--s3_bbox_root', default='', type=str)
    parser.add_argument('--bbox_thredshold', default=0.0, type=float)
    parser.add_argument('--use_bbox_caption_aug', default=False, type=bool)
    parser.add_argument('--bbox_class_dict_path', default='', type=str)
    parser.add_argument('--use_feature_dataset', default=False, type=bool)
    parser.add_argument('--need_vae', default=True, type=bool)
    parser.add_argument('--use_blip_caption', default=True, type=bool)
    parser.add_argument('--blip_caption_root', default='', type=str)
    parser.add_argument('--blip_caption_p', default=1.0, type=float)
    parser.add_argument('--s3_blip_caption_root', default='', type=str)
    parser.add_argument('--resize_latent', default=False, type=bool)
    parser.add_argument('--resize_latent_p', default=0.5, type=float)
    parser.add_argument('--segmentation_label_root', default='', type=str)
    parser.add_argument('--s3_segmentation_label_root', default='', type=str)
    parser.add_argument('--mask_padding', default=10, type=int)
    parser.add_argument('--cross_weight', default=1.0, type=float)
    parser.add_argument('--time_st', default=0, type=int)
    parser.add_argument('--time_ed', default=1000, type=int)
    parser.add_argument('--lmdb2json_ratio', default=100, type=int)
    parser.add_argument('--limit_dataloader_len', default=-1, type=int)
    parser.add_argument('--use_cross_uncond', default=False, type=bool)
    parser.add_argument('--cross_uncond_p', default=0.3, type=float)
    parser.add_argument('--attn_mask_neg_inf', default=False, type=bool)
    parser.add_argument('--attn_mask_softmax', default=False, type=bool)
    parser.add_argument('--attn_mask_amplify', default=1.0, type=float)
    parser.add_argument('--save_epoch_freq', default=10, type=int)
    parser.add_argument('--phase_num', default=5, type=int)
    parser.add_argument('--drop_desc', default=False, type=bool)
    parser.add_argument('--append_uncond', default=False, type=bool)
    parser.add_argument('--mask_area_threshold', default=0.0, type=float)
    # parser.add_argument('--dilate_mask', default=False, type=bool)
    # parser.add_argument('--dilate_kernel', default=3, type=int)
    # parser.add_argument('--dilate_iter', default=1, type=int)
    parser.add_argument('--fuser_mean_loss_weight', default=1.0, type=float)
    parser.add_argument('--unify_seed', default=False, type=bool)
    parser.add_argument('--global_seed', default=0, type=int)
    parser.add_argument('--only_use_one_box', default=False, type=bool)
    parser.add_argument('--desc_use_sup_mask', default=False, type=bool)
    parser.add_argument('--swap_desc_with_null', default=False, type=bool)
    parser.add_argument('--swap_desc_with_null_p', default=0.0, type=float)
    parser.add_argument('--null_use_one_box', default=False, type=bool)
    parser.add_argument('--phase_random_order', default=False, type=bool)
    parser.add_argument('--phase_random_order_v2', default=False, type=bool)
    parser.add_argument('--one_phase_one_instance', default=False, type=bool)
    parser.add_argument('--phase_random_before_anything', default=False, type=bool)
    parser.add_argument('--use_bbox_loss_v1', default=False, type=bool)
    parser.add_argument('--bbox_loss_v1_weight', default=1.0, type=float)
    parser.add_argument('--phase_random_order_v3', default=False, type=bool)
    parser.add_argument('--bbox_mask_dilation', default=False, type=bool)
    parser.add_argument('--phase_random_order_v4', default=False, type=bool)
    parser.add_argument('--cat_small_size', default=False, type=bool)
    parser.add_argument('--input_with_box', default=False, type=bool)
    parser.add_argument('--use_aug_attn1_loss_v1', default=False, type=bool)
    parser.add_argument('--aug_attn1_loss_v1_weight', default=1.0, type=float)
    parser.add_argument('--use_aug_attn1_loss_v2', default=False, type=bool)
    parser.add_argument('--aug_attn1_loss_v2_weight', default=1.0, type=float)
    parser.add_argument('--use_aug_attn1_loss_v3', default=False, type=bool)
    parser.add_argument('--aug_attn1_loss_v3_weight', default=1.0, type=float)
    parser.add_argument('--use_pos_attn1_loss_v1', default=False, type=bool)
    parser.add_argument('--pos_attn1_loss_v1_weight', default=1.0, type=float)
    parser.add_argument('--pos_attn1_loss_layers', default=[0, 1, 2, 3], type=list)
    parser.add_argument('--use_pos_attn1_loss_v2', default=False, type=bool)
    parser.add_argument('--pos_attn1_loss_v2_weight', default=1.0, type=float)
    parser.add_argument('--different_t_in_one_batch', default=False, type=bool)
    parser.add_argument('--phase_random_order_v3_p', default=1.0, type=float)
    parser.add_argument('--need_mox_pretrained', default=False, type=bool)
    parser.add_argument('--mox_pretrained_path', default='', type=str)
    parser.add_argument('--use_limit_instance_num', default=False, type=bool)
    parser.add_argument('--limit_instance_num', default=100, type=int)
    parser.add_argument('--replace_desc_with_phase', default=False, type=bool)
    parser.add_argument('--replace_desc_with_phase_p', default=1.0, type=float)
    parser.add_argument('--aug_phase_with_and', default=False, type=bool)
    parser.add_argument('--use_fuser_desc_mean_loss', default=False, type=bool)
    parser.add_argument('--fuser_desc_mean_loss_weight', default=1.0, type=float)
    parser.add_argument('--use_fuser_desc_mean_loss_v2', default=False, type=bool)
    parser.add_argument('--fuser_desc_mean_loss_weight_v2', default=1.0, type=float)
    parser.add_argument('--use_pre_attn_loss', default=False, type=bool)
    parser.add_argument('--pre_attn_loss_weight', default=1.0, type=float)
    parser.add_argument('--pre_attn_loss_align_max1', default=False, type=bool)
    parser.add_argument('--pre_attn_loss_cal_map', default=False, type=bool)
    parser.add_argument('--pre_attn_loss_with_mask', default=False, type=bool)
    parser.add_argument('--inter_mode', default='nearest', type=str)
    parser.add_argument('--pre_attn_loss_strict_mask', default=False, type=bool)
    parser.add_argument('--pre_attn_loss_ave_heads', default=False, type=bool)
    parser.add_argument('--pre_attn_loss_ave_layer', default=False, type=bool)
    parser.add_argument('--pre_attn_loss_only_instance', default=False, type=bool)
    parser.add_argument('--use_pre_attn_fore_max_loss', default=False, type=bool)
    parser.add_argument('--pre_attn_fore_max_loss_weight', default=1.0, type=float)
    parser.add_argument('--pre_attn_fore_max_detach', default=False, type=bool)
    parser.add_argument('--use_fore_enh_loss', default=False, type=bool)
    parser.add_argument('--fore_enh_loss_weight', default=1.0, type=float)
    parser.add_argument('--replace_desc_with_phase_before_fuser', default=False, type=bool)
    parser.add_argument('--replace_desc_with_phase_before_fuser_p', default=1.0, type=float)
    parser.add_argument('--only_train_MuViC_partly', default=False, type=bool)
    parser.add_argument('--train_MuViC_partly_name', default=[], type=list)
    parser.add_argument('--erase_partly_name', default=True, type=bool)
    parser.add_argument('--CA_inner_MuViC', default=False, type=bool)
    parser.add_argument('--load_RGB', default=False, type=bool)
    parser.add_argument('--RGB_path', default='../dataset/000000_depth', type=str)
    parser.add_argument('--s3_RGB_path', default='', type=str)
    parser.add_argument('--instance_score_threshold', default=0.0, type=float)
    parser.add_argument('--need_dino', default=False, type=bool)
    parser.add_argument('--img_instance_num', default=2, type=int)
    parser.add_argument('--use_local_img_feature', default=False, type=bool)
    parser.add_argument('--uniform_aug_phase_with_and', default=False, type=bool)
    parser.add_argument('--drop_phase_in_img', default=False, type=bool)
    parser.add_argument('--drop_phase_in_img_p', default=0.5, type=float)
    parser.add_argument('--replace_desc_with_null', default=False, type=bool)
    parser.add_argument('--replace_desc_with_null_p', default=0, type=float)
    parser.add_argument('--use_multi_res_noise', default=False, type=bool)
    parser.add_argument('--multi_res_noise_scale', default=0.5, type=float)
    parser.add_argument('--use_pyramid_noise', default=False, type=bool)
    parser.add_argument('--pyramid_noise_scale', default=0.5, type=float)
    parser.add_argument('--need_unet_pretrained', default=False, type=bool)
    parser.add_argument('--unet_pretrained_path', default='', type=str)
    parser.add_argument('--need_MIGC_pretrained', default=False, type=bool)
    parser.add_argument('--MIGC_preserve', default=['ea', 'norm', 'la', 'sac', 'pos_net', 'gamma'], type=list)
    parser.add_argument('--MIGC_train', default=['ea', 'norm', 'la', 'sac', 'pos_net', 'gamma'], type=list)
    parser.add_argument('--SD_depth_frozen', default=[], type=list)
    parser.add_argument('--copy_channel', default=False, type=bool)
    parser.add_argument('--lmdb_suffix', default='_laion_ldm3d', type=str)
    parser.add_argument('--unet_path', default='/mnt/data1/dewei/ckpt/new_sd14', type=str)
    parser.add_argument('--depth_state_dict', default=None, type=str)
    parser.add_argument('--use_laion', default=False, type=bool)
    parser.add_argument('--use_laion_clean', default=False, type=bool)
    parser.add_argument('--depth_migc_base_model', default='../../../ckpt/ldm3d-4c', type=str)
    parser.add_argument('--del_color', default=False, type=bool)
    parser.add_argument('--dilate_mask', default=False, type=bool)
    parser.add_argument('--dilate_kernel', default=15, type=int)
    parser.add_argument('--dilate_p', default=0.5, type=float)
    parser.add_argument('--use_json_v2', default=False, type=bool)
    parser.add_argument('--replace_phase_with_null', default=False, type=bool)
    parser.add_argument('--min_edge_limit', default=-1, type=int)
    # min_edge_limit
    
    def parse_int_or_list(value):
        try:
            return int(value)
        except ValueError:
            return list(map(int, value.split(',')))
        
    parser.add_argument('--dilate_iter', default=5, type=parse_int_or_list)
    '''
    
                                uniform_aug_phase_with_and=cfg.uniform_aug_phase_with_and,
                                drop_phase_in_img=cfg.drop_phase_in_img,
                                drop_phase_in_img_p=cfg.drop_phase_in_img_p,
                                replace_desc_with_null=cfg.replace_desc_with_null,
                                replace_desc_with_null_p=cfg.replace_desc_with_null_p,
    '''
    
    # diffusers
    
    parser.add_argument('--migc_type', default='migc', type=str)





    args = parser.parse_args()
    args = merge_args(args, args.yml_path)
    # env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    # if env_local_rank != -1 and env_local_rank != args.local_rank:
    #     args.local_rank = env_local_rank
    # if started with torch.distributed.launch
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ.get('LOCAL_RANK'))

    # use remote paths specified in config
    if mox is not None:
        args.list_data_root = args.cloud_list_data_root
        args.img_data_root = args.cloud_img_data_root

    if args.input_act_ckpt is not None and ',' in args.input_act_ckpt:
        args.input_act_ckpt = list_from_str(args.input_act_ckpt)
        assert isinstance(args.input_act_ckpt, list) and len(args.input_act_ckpt) == 2
    if args.output_act_ckpt is not None and ',' in args.output_act_ckpt:
        args.output_act_ckpt = list_from_str(args.output_act_ckpt)
        assert isinstance(args.output_act_ckpt, list) and len(args.output_act_ckpt) == 2
    return args

def _check_dir(dist_dir):
    copy_flag = True
    if os.path.exists(dist_dir):
        copy_flag = False
    if not os.path.exists(os.path.dirname(dist_dir)):
        os.makedirs(os.path.dirname(dist_dir))
    return copy_flag


def cmd_exec(cmd, just_print=False):
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("\n{}:INFO:{}".format(t, cmd))
    if not just_print:
        os.system(cmd)

def mox_copy(src, dst, parallel=False):
    while True:
        failed = 0
        try:
            if parallel:
                mox.file.copy_parallel(src, dst)
            else:
                mox.file.copy(src, dst)
            break
        except Exception as e:
            failed += 1
            time.sleep(60)
            if failed % 10 == 0:
                print("error, maybe need check. copy failed {} times from {} to {}".format(failed, src, dst))
                print("error message: {}".format(e))

def uncompress(tar_file):
    ret = subprocess.check_output("ps -ef | grep tar | grep xf | grep -v grep | grep -v \'sh -c\' | wc -l", shell=True)
    ret = int(ret.decode('utf-8'))
    if ret > 0:
        cmd_exec("find uncompress running process:", just_print=True)
        os.system("ps -ef | grep tar | grep xf | grep -v grep | grep -v \'sh -c\'")
    tar_name = os.path.split(tar_file)[-1]
    tar_dir = os.path.dirname(tar_file)
    cmd_exec('cd {} && tar -xf {} && rm -rf {} &'.format(tar_dir, tar_name, tar_name))

def copy_data_to_cache(src_dir='', dist_dir='', rank=0, world_size=1, args=None):
    start_t = time.time()
    copy_flag = _check_dir(dist_dir)

    if args is not None and args.local_shuffle_type == 4:
        local_shuffle = args.local_shuffle_type
        zip_max_split = args.zip_max_split
        print("training in cloud, using local_shuffle_type={}, zip_max_split={}".format(args.local_shuffle_type, zip_max_split))
    else:
        local_shuffle = 0
        zip_max_split = -1

    if copy_flag:
        print('copy from {} to {}'.format(src_dir, dist_dir))
        tar_files = []
        t0 = time.time()
        if ".mindrecord" in src_dir:
            src_dir = os.path.split(src_dir)[0]
            dist_dir = os.path.split(dist_dir)[0]

        last_file = None
        allready_uncompress = []
        copy_dir = []

        if mox.file.is_directory(src_dir): # no new tar in tar !!!
            subfiles = [subfile for subfile in mox.file.list_directory(src_dir, recursive=False)]
            # if 'LAION' in src_dir:
            #     subfiles = subfiles[:args.tar_max_split]
            print('SUBFILES: ', subfiles)
            subfiles.sort()
            for subfile in subfiles:
                if subfile in ['coco_val.tar', 'imagenet.tar', 'test_data.tar']:
                    continue
                sub_src_dir = os.path.join(src_dir, subfile)
                sub_dist_dir = os.path.join(dist_dir, subfile)

                if local_shuffle and "split_part" in sub_src_dir and "map" not in sub_src_dir:  # not include "AAAA_split_partBBBB_map.pkl"
                    part_idx = int(os.path.split(sub_src_dir)[-1][-8:-4]) # "AAAA_split_partBBBB.pkl" or "AAAA_split_partBBBB.zip"
                    if part_idx % world_size != rank or part_idx >= zip_max_split:
                        continue

                if local_shuffle and "map" in sub_src_dir:
                    part_idx = int(os.path.split(sub_src_dir)[-1][-12:-8]) # not include "AAAA_split_partBBBB_map.pkl"
                    if part_idx % world_size != rank or part_idx >= zip_max_split:
                        continue

                # uncompress last file
                if last_file is not None and last_file.endswith('.tar'):
                    uncompress(last_file)
                    allready_uncompress.append(last_file)

                # copy new file
                cmd_exec('copy from {} to {}'.format(sub_src_dir, sub_dist_dir), just_print=True)
                if mox.file.is_directory(sub_src_dir):
                    mox_copy(sub_src_dir, sub_dist_dir, parallel=True)
                    copy_dir.append(sub_dist_dir)
                    last_file = None
                else:
                    mox_copy(sub_src_dir, sub_dist_dir)
                    last_file = sub_dist_dir

            if last_file is not None and last_file.endswith('.tar'):
                uncompress(last_file)
                allready_uncompress.append(last_file)

        else:
            mox_copy(src_dir, dist_dir)
            if dist_dir.endswith('tar') or dist_dir.endswith('tar.gz'):
                tar_files.append(dist_dir)

        t1 = time.time()
        cmd_exec('copy datasets, time used={:.2f}s'.format(t1 - t0), just_print=True)

        # final check, no tar forget
        for _dir in copy_dir:
            tar_list = list(Path(_dir).glob('**/*.tar'))
            tar_files.extend(tar_list)
            tar_list = list(Path(_dir).glob('**/*.tar.gz'))
            tar_files.extend(tar_list)

        tar_files = [x for x in tar_files if str(x) not in allready_uncompress]

        cmd_exec('tar_files:{}'.format(tar_files), just_print=True)
        for tar_file in tar_files:
            tar_dir = os.path.dirname(tar_file)
            cmd_exec('cd {} && tar -xf {} && rm -rf {} &'.format(tar_dir, tar_file, tar_file))

        # final check, no tar process
        while True:
            ret = subprocess.check_output("ps -ef | grep tar | grep xf | grep -v grep | grep -v \'sh -c\' | wc -l", shell=True)
            ret = int(ret.decode('utf-8'))
            if ret == 0:
                cmd_exec("not find tar process, break", just_print=True)
                break
            else:
                cmd_exec("find {} tar process, sleep 10s".format(ret), just_print=True)
                os.system("ps -ef | grep tar | grep xf | grep -v grep | grep -v \'sh -c\'")
            time.sleep(10)

        cmd_exec('copy data completed!', just_print=True)

        # cmd_exec("since data already exists, copying is not required, src={}, dst={}".format(src_dir, dist_dir), just_print=True)

    end_t = time.time()
    cmd_exec('copy cost total time {:.2f} sec'.format(end_t - start_t), just_print=True)

class ConfigObject:
    def __init__(self, entries):
        for a, b in entries.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [ConfigObject(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, ConfigObject(b) if isinstance(b, dict) else b)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()

def parse_yaml(fp):
    with open(fp, 'r') as fd:
        cont = fd.read()
        try:
            y = yaml.load(cont, Loader=yaml.FullLoader)
        except:
            y = yaml.load(cont)
        return y

def parse_replace_roma(fp, copy_to_cache=False, rank=0, world_size=1, untar=True, args=None):
    y = parse_yaml(fp)
    for y_key in y.keys():
        y_val = y[y_key]
        if mox is not None and isinstance(y_val, str) and y_val.startswith('s3://'):
            # copy to /cache and replace to /cache
            y_val_cache = y_val.replace('s3://', '/cache/')
            if copy_to_cache and 'LAION' not in y_val:
                # rank and world_size means node rank and node world_size
                # ugly for rec with idx
                if y_val_cache.endswith('.rec'):
                    y_val_rec_idx = y_val.replace('.rec', '.idx')
                    y_val_cache_rec_idx = y_val_cache.replace('.rec', '.idx')
                    print('copy {} to {}'.format(y_val_rec_idx,y_val_cache_rec_idx))
                    copy_data_to_cache(y_val_rec_idx, y_val_cache_rec_idx,
                                       rank=rank,
                                       world_size=world_size,
                                       args=args)
                copy_data_to_cache(y_val, y_val_cache,
                                   rank=rank,
                                   world_size=world_size,
                                   args=args)
            if copy_to_cache and 'LAION' in y_val:
                copy_data_to_cache(y_val, y_val_cache,
                                   rank=rank,
                                   world_size=world_size,
                                   args=args)
            y[y_key] = y_val_cache
    return y

def merge_args(args, args_yml_fn):
    if os.path.exists(args_yml_fn):
        args_dict = args.__dict__
        args_yml = parse_replace_roma(args_yml_fn, copy_to_cache=False)
        args_dict_merge = dict(args_dict, **args_yml)
        args = ConfigObject(args_dict_merge)
    elif len(args_yml_fn) != 0:
        print('yml file {} is not existed'.format(args_yml_fn))
        exit(0)

    sys_args = sys.argv[1:]
    for arg in sys_args:
        if re.match('^--(.*)=(.*)$', arg):
            arg = arg.replace('--', '')
            key, val = arg.split('=')
            default_value = getattr(args, key)
            new_value = type(default_value)(val)
            if default_value != new_value:
                print('set {} from {} to {}'.format(key, default_value ,new_value))
                setattr(args, key, new_value)
        else:
            print('unmatched, arg: {}'.format(arg))
    return args