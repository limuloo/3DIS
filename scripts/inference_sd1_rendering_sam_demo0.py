'''
CUDA_VISIBLE_DEVICES=4 python inferencev2_mig_controlnet_benchmark.py --out_path=exp003/MIGC_ldm3d_121_rgb --model_path=pretrained_weights/MIGC_SD14.ckpt --depth_path=exp003/MIGC_ldm3d_121 --use_migc_v1

'''
import yaml
from diffusers import ControlNetModel, EulerDiscreteScheduler, UniPCMultistepScheduler
from threeDIS.utils import seed_everything
from threeDIS.pipeline_sd1_rendering import SD1RenderingPipeline, AttentionStore
from threeDIS.detail_renderer.detail_renderer_sd1 import DetailRendererSD1
import os 
from tqdm import tqdm
from PIL import Image
import argparse
import cv2  
import torch
import numpy as np
from copy import deepcopy
import random
from threeDIS.utils import load_sd1_renderer, display_instance_with_masks

from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2

parser = argparse.ArgumentParser(description='MIGC Inference')
parser.add_argument('--fft', action='store_true')
parser.add_argument('--fft_mid', type=bool, default=True)
parser.add_argument('--fft_up', type=bool, default=True)
parser.add_argument('--fft_up_res', type=int, default=64)
parser.add_argument('--control_CN', action='store_true')
parser.add_argument('--height', type=int, default=512)
parser.add_argument('--width', type=int, default=512)
parser.add_argument('--fft_d', type=int, default=8)
args = parser.parse_args()


import numpy as np



if __name__ == '__main__':

    # Construct SAM-2 predictor
    sam2_checkpoint = "pretrained_weights/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    predictor = build_sam2(model_cfg, sam2_checkpoint, device='cuda')
    predictor = SAM2ImagePredictor(predictor)

    
    # Construct ControlNet
    controlnet_path = "lllyasviel/sd-controlnet-depth" if not os.path.isdir("/mnt/sda/zdw/ckpt/sd-controlnet-depth") else "/mnt/sda/zdw/ckpt/sd-controlnet-depth"
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
    if args.control_CN:
        load_sd1_renderer(controlnet, attn_processor=DetailRendererSD1)
    controlnet = controlnet.to("cuda")

    # Construct SD1 rendering pipeline
    sd1x_path = "CompVis/stable-diffusion-v1-4" if not os.path.isdir("/mnt/sda/zdw/ckpt/new_sd14") else "/mnt/sda/zdw/ckpt/new_sd14"
    pipe = SD1RenderingPipeline.from_pretrained(sd1x_path, controlnet=controlnet, torch_dtype=torch.float16)
    # You can go to https://civitai.com/search/models?baseModel=SD%201.4&baseModel=SD%201.5&sortBy=models_v5 find a base model with better generation ability to achieve better creations.
    # pipe = SD1RenderingPipeline.from_single_file('/home/zdw/project/MIGC/migc_gui_weights/sd/realisticVisionV60B1_v51HyperVAE.safetensors', controlnet=controlnet, torch_dtype=torch.float16)
    # pipe = SD1RenderingPipeline.from_single_file('/home/zdw/project/MIGC/migc_gui_weights/sd/cetusMix_Whalefall2.safetensors', controlnet=controlnet, torch_dtype=torch.float16)

    # load lora, optional
    # pipe.load_lora_weights("/mnt/sda/zdw/project/opensource/3DIS/pretrained_weights/xrs2.0.safetensors", scale=0.9)
    
    # load sd1 renderer
    load_sd1_renderer(pipe.unet, attn_processor=DetailRendererSD1)
    pipe = pipe.to("cuda")

    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    seed = 42
    
    prompt_final = [['bestquality, detailed, 8k.a photo of a black potted plant and a yellow refrigerator and a brown surfboard', 'a black potted plant', 'a brown surfboard', 'a yellow refrigerator']]
    bboxes = [[[0.5717187499999999, 0.0, 0.8179531250000001, 0.29807511737089204], [0.85775, 0.058755868544600943, 0.9991875, 0.646525821596244], [0.6041562500000001, 0.284906103286385, 0.799046875, 0.9898591549295774]]]

    depth_path = './data/room1_depth.png'
    instance_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

    # SAM-Enhance Layout
    instance_img_rgb = cv2.imread(depth_path, cv2.IMREAD_COLOR)
    predictor.set_image(instance_img_rgb)
    sam_bboxes = []
    for bbox in bboxes[0]:
        sam_bboxes.append([int(i * 512) for i in bbox]) 
    input_box = np.array(sam_bboxes)
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )
    masks = masks.astype(np.float32)
    mask_img = display_instance_with_masks(instance_img_rgb,
                                            masks, prompt_final)
    cv2.imwrite('./sam_layout_vis.png', mask_img)
    instance_img = Image.fromarray(instance_img)

    seed_everything(seed)
    image = pipe(deepcopy(prompt_final), bboxes, num_inference_steps=50, 
                    guidance_scale=7.5, 
                    ControlSteps=50, image=instance_img,  controlnet_conditioning_scale=1.0
                    ,aug_phase_with_and=True  , control_fft_filter=args.fft, sam_masks=masks,
                    control_CN=args.control_CN, fft_up=args.fft_up, fft_mid=args.fft_mid, fft_up_res=int(args.fft_up_res * args.height / 512),
                    height=args.height, width=args.width, fft_d=args.fft_d).images[0]
        
    image.save(os.path.join('./', "sd15_room_rendering_sam_enhance.png"))