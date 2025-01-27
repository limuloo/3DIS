import argparse
import numpy as np
import torch
import os
import yaml
import random
from diffusers.utils.import_utils import is_accelerate_available
from diffusers.pipelines import StableDiffusionXLPipeline
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from threeDIS.pipeline_stable_diffusion_layout2depth import StableDiffusionL2DPipeline, MIGCProcessor
from diffusers import EulerDiscreteScheduler
if is_accelerate_available():
    from accelerate import init_empty_weights
from contextlib import nullcontext


def seed_everything(seed):
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


import torch
from typing import Callable, Dict, List, Optional, Union
from collections import defaultdict


# We need to set Attention Processors for the following keys.
all_processor_keys = [
    'down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor', 'down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor',
    'down_blocks.0.attentions.1.transformer_blocks.0.attn1.processor', 'down_blocks.0.attentions.1.transformer_blocks.0.attn2.processor',
    'down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor', 'down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor',
    'down_blocks.1.attentions.1.transformer_blocks.0.attn1.processor', 'down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor',
    'down_blocks.2.attentions.0.transformer_blocks.0.attn1.processor', 'down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor',
    'down_blocks.2.attentions.1.transformer_blocks.0.attn1.processor', 'down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor',
    'up_blocks.1.attentions.0.transformer_blocks.0.attn1.processor', 'up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor',
    'up_blocks.1.attentions.1.transformer_blocks.0.attn1.processor', 'up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor',
    'up_blocks.1.attentions.2.transformer_blocks.0.attn1.processor', 'up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor',
    'up_blocks.2.attentions.0.transformer_blocks.0.attn1.processor', 'up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor',
    'up_blocks.2.attentions.1.transformer_blocks.0.attn1.processor', 'up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor',
    'up_blocks.2.attentions.2.transformer_blocks.0.attn1.processor', 'up_blocks.2.attentions.2.transformer_blocks.0.attn2.processor',
    'up_blocks.3.attentions.0.transformer_blocks.0.attn1.processor', 'up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor',
    'up_blocks.3.attentions.1.transformer_blocks.0.attn1.processor', 'up_blocks.3.attentions.1.transformer_blocks.0.attn2.processor',
    'up_blocks.3.attentions.2.transformer_blocks.0.attn1.processor', 'up_blocks.3.attentions.2.transformer_blocks.0.attn2.processor',
    'mid_block.attentions.0.transformer_blocks.0.attn1.processor', 'mid_block.attentions.0.transformer_blocks.0.attn2.processor'
]


def load_migc(unet, pretrained_MIGC_path: Union[str, Dict[str, torch.Tensor]], attn_processor, strict=True,
                     **kwargs):

    state_dict = torch.load(pretrained_MIGC_path, map_location="cpu")

    attn_processors = {}

    adapter_grouped_dict = defaultdict(dict)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    for key, value in state_dict.items():
        attn_processor_key, sub_key = key.split('.attn2.processor.')
        adapter_grouped_dict[attn_processor_key][sub_key] = value

    # Create MIGC Processor
    config = {'not_use_migc': False}
    for key, value_dict in adapter_grouped_dict.items():
        dim = value_dict['migc.norm.bias'].shape[0]
        config['C'] = dim
        key_final = key + '.attn2.processor'
        if key_final.startswith("mid_block"):
            place_in_unet = "mid"
        elif key_final.startswith("up_blocks"):
            place_in_unet = "up"
        elif key_final.startswith("down_blocks"):
            place_in_unet = "down"
        attn_processors[key_final] = attn_processor(config, place_in_unet)
        attn_processors[key_final].load_state_dict(value_dict, strict=strict)
        attn_processors[key_final].to(device=unet.device, dtype=unet.dtype)

    # Create CrossAttention/SelfAttention Processor
    config = {'not_use_migc': True}
    for key in all_processor_keys:
        if key not in attn_processors.keys():
            if key.startswith("mid_block"):
                place_in_unet = "mid"
            elif key.startswith("up_blocks"):
                place_in_unet = "up"
            elif key.startswith("down_blocks"):
                place_in_unet = "down"
            attn_processors[key] = attn_processor(config, place_in_unet, migc_type='no_migc')
    unet.set_attn_processor(attn_processors)


def load_sd1_renderer(unet, attn_processor, strict=True,
                        **kwargs):


    all_processor_keys = set()
    for unet_key in unet.state_dict().keys():
        if '.attn1.' in unet_key:
            all_processor_keys.add(unet_key.split('.attn1.')[0] + '.attn1.processor')
        if '.attn2.' in unet_key:
            all_processor_keys.add(unet_key.split('.attn2.')[0] + '.attn2.processor')
    all_processor_keys = list(all_processor_keys)

    
    # Create CrossAttention/SelfAttention Processor
    attn_processors = {}
    config = {'not_use_migc': True}
    for key in all_processor_keys:
        if key.startswith("mid_block"):
            place_in_unet = "mid"
        elif key.startswith("up_blocks"):
            place_in_unet = "up"
        elif key.startswith("down_blocks"):
            place_in_unet = "down"
        attn_processors[key] = attn_processor(config, place_in_unet)
    unet.set_attn_processor(attn_processors)


def get_all_processor_keys(model, parent_name=''):
    all_processor_keys = []
    
    for name, module in model.named_children():
        full_name = f'{parent_name}.{name}' if parent_name else name
        
        # Check if the module has 'processor' attribute
        if hasattr(module, 'processor'):
            all_processor_keys.append(f'{full_name}.processor')
            # print(type(module.processor))
        
        # Recursively check submodules
        all_processor_keys.extend(get_all_processor_keys(module, full_name))
    
    return all_processor_keys

    


from diffusers import UniPCMultistepScheduler

def ConstructLayoutToDepthPipe():
    project_path = os.path.dirname(os.path.dirname(__file__))

    sd_path = "Intel/ldm3d-4c" if not os.path.isdir('/mnt/sda/zdw/ckpt/ldm3d-4c') else '/mnt/sda/zdw/ckpt/ldm3d-4c'
    pipe = StableDiffusionL2DPipeline.from_pretrained(sd_path)
    
    # load fine-tuned Text-to-Depth model
    unet_path = f'{project_path}/pretrained_weights/unet_0901.ckpt'
    assert os.path.isfile(unet_path), f"Checkpoint file not found at {unet_path}"
    pipe.unet.load_state_dict(torch.load(unet_path))

    # load Layout-to-Depth Adapter
    layout_adapter_ckpt_path = f'{project_path}/pretrained_weights/layout_adapter.ckpt'
    assert os.path.isfile(layout_adapter_ckpt_path), f"Checkpoint file not found at {layout_adapter_ckpt_path}"
    load_migc(pipe.unet, layout_adapter_ckpt_path, attn_processor=MIGCProcessor)
    
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe = pipe.to("cuda")
    return pipe


from threeDIS.pipeline_flux_rendering import FluxRenderingPipeline
from threeDIS.detail_renderer.detail_renderer_flux import DetailRendererFLUX

def ConstructFluxRenderingPipeline():

    flux_depth_path = "black-forest-labs/FLUX.1-Depth-dev" if not os.path.isdir("/mnt/sda/zdw/ckpt/FLUX.1-Depth-dev") else "/mnt/sda/zdw/ckpt/FLUX.1-Depth-dev"
    
    pipe = FluxRenderingPipeline.from_pretrained(flux_depth_path, torch_dtype=torch.bfloat16).to("cuda")
    all_processor_keys_flux = get_all_processor_keys(pipe.transformer)
    attn_processors = {}
    for key in all_processor_keys_flux:
        attn_processors[key] = DetailRendererFLUX()
    pipe.transformer.set_attn_processor(attn_processors)

    pipe = pipe.to("cuda")
    return pipe


def ConstructSAM2Predictor():
    project_path = os.path.dirname(os.path.dirname(__file__))
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.build_sam import build_sam2
    sam2_checkpoint = f"{project_path}/pretrained_weights/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    sam_predictor = build_sam2(model_cfg, sam2_checkpoint, device='cuda')
    sam_predictor = SAM2ImagePredictor(sam_predictor)
    return sam_predictor


def get_sup_mask(mask_list):
    or_mask = np.zeros_like(mask_list[0])
    for mask in mask_list:
        or_mask += mask
    or_mask[or_mask >= 1] = 1
    sup_mask = 1 - or_mask
    return sup_mask


def construct_supplement_mask_from_layout(bbox_list, height, width):
    import math
    guidance_masks = []
    for bbox in bbox_list:  
        guidance_mask = np.zeros((height, width))
        w_min = int(width * bbox[0])
        w_max = int(math.ceil(width * bbox[2]))
        h_min = int(height * bbox[1])
        h_max = int(math.ceil(height * bbox[3]))
        guidance_mask[h_min: h_max, w_min: w_max] = 1.0
        guidance_masks.append(guidance_mask[None, ...])
    
    sup_mask = get_sup_mask(guidance_masks)
    supplement_mask = torch.from_numpy(sup_mask[None, ...])
    return supplement_mask


def display_instance_with_masks(instance_img_rgb, masks, prompt):
    """
    instance_img_rgb: (512, 512, 3) RGB instance image
    masks: (N, 1, 512, 512) binary masks (0 or 1)
    """
    # Ensure masks are converted to the right shape (N, 512, 512)
    masks = masks.squeeze(1)

    # Generate random colors for each mask (N, 3)
    N = masks.shape[0]
    # colors = np.random.rand(N, 3)
    color_dict = {
        'black': [0.0, 0.0, 0.0],
        'white': [255.0, 255.0, 255.0],
        'red': [255.0, 0.0, 0.0],
        'green': [0.0, 255.0, 0.0],
        'blue': [0.0, 0.0, 255.0],
        'yellow': [255.0, 255.0, 0.0],
        'purple': [255.0, 0.0, 255.0],
        'pink': [255.0, 192.0, 203.0],
        'brown': [165.0, 42.0, 42.0],
        'gray': [128.0, 128.0, 128.0],
        'grey': [128.0, 128.0, 128.0],
        'orange': [255.0, 165.0, 0.0],
        
    }
    colors = []
    for i in range(N):
        flag = False
        for color in color_dict:
            if color in prompt[0][i + 1]:
                now_color = np.array(color_dict[color])
                now_color[0], now_color[2] = now_color[2], now_color[0]
                colors.append(now_color)
                flag = True
                break
        if not flag:
            colors.append(np.random.rand(3) * 255.0)

    # Create a copy of the instance image to overlay masks
    overlay_image = instance_img_rgb.copy()

    # Iterate over each mask and apply color
    for i in range(N):
        mask = masks[i]
        color = colors[i]

        # Expand mask to 3 channels
        mask_rgb = np.stack([mask] * 3, axis=-1)

        # Apply the color to the mask area
        overlay_image = np.where(mask_rgb, overlay_image * 0.5 + color * 0.5, overlay_image)

    return overlay_image