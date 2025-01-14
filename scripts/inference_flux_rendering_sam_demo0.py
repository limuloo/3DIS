import torch
from threeDIS.pipeline_flux_rendering import FluxRenderingPipeline
from PIL import Image
import numpy as np
import argparse
from huggingface_hub import login
from threeDIS.detail_renderer.detail_renderer_flux import DetailRendererFLUX
from threeDIS.utils import get_all_processor_keys
import os




parser = argparse.ArgumentParser(description='MIGC Inference')
parser.add_argument('--out_path', type=str, default='/data/hpfs/zdw/code/RD/3dis_v2/teaser4/output', help='output directory')
parser.add_argument('--depth_path', type=str, default='/data/hpfs/zdw/code/RD/3dis_v2/teaser4/depth', help='path to MIGC checkpoint')
parser.add_argument('--rgb_path', type=str, default='/data/hpfs/zdw/code/RD/3dis_v2/teaser4/depth', help='path to MIGC checkpoint')
# parser.add_argument('--instance_token_num', type=int, default=50)
parser.add_argument('--l', type=int, default=0)
parser.add_argument('--r', type=int, default=799)
parser.add_argument('--height', type=int, default=512)
parser.add_argument('--width', type=int, default=512)
parser.add_argument('--hard_control_steps', type=int, default=20)
parser.add_argument('--i2i', type=int, default=4, help='I2I_control_steps')
parser.add_argument('--i2t', type=int, default=20, help='I2T_control_steps')
parser.add_argument('--t2i', type=int, default=20, help='T2I_control_steps')
parser.add_argument('--t2t', type=int, default=20, help='T2T_control_steps')
parser.add_argument('--multi_seed', action='store_true')
parser.add_argument('--use_sam_enhance', action='store_true')

args = parser.parse_args()




if __name__ == '__main__':
    seed = 7505876460391471034

    # https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev
    flux_depth_path = "black-forest-labs/FLUX.1-Depth-dev" if not os.path.isdir("/mnt/sda/zdw/ckpt/FLUX.1-Depth-dev") else "/mnt/sda/zdw/ckpt/FLUX.1-Depth-dev"
    
    pipe = FluxRenderingPipeline.from_pretrained(flux_depth_path, torch_dtype=torch.bfloat16).to("cuda")
    all_processor_keys_flux = get_all_processor_keys(pipe.transformer)
    attn_processors = {}
    for key in all_processor_keys_flux:
        attn_processors[key] = DetailRendererFLUX()
    pipe.transformer.set_attn_processor(attn_processors)

    if args.use_sam_enhance:
        # Construct SAM Predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.build_sam import build_sam2
        sam2_checkpoint = "./pretrained_weights/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"

        sam_predictor = build_sam2(model_cfg, sam2_checkpoint, device='cuda')
        sam_predictor = SAM2ImagePredictor(sam_predictor)


    img_prompt = "a beautiful chinese girl is reading a book"
    instance_prompts = [["a book", 'a golden table', 'a white lamp', 'a beautiful young chinese girl, student uniform', 'black silk stockings',
                         "a wooden bed", "black long boot", "black long boot", "a wooden bed"]]
    for instance_prompt in instance_prompts[0]:
        img_prompt += ',' + instance_prompt


    instance_box_list = [[0.3359375, 0.3515625, 0.6380208333333334, 0.5234375], [0.765625, 0.328125, 0.9947916666666666, 0.875], 
                         [0.7578125, 0.0234375, 0.9869791666666666, 0.3984375], [0.34375, 0.046875, 0.6354166666666666, 0.40625],
                         [0.3229166666666667, 0.4765625, 0.6458333333333334, 0.984375], [0.052083333333333336, 0.1640625, 0.7708333333333334, 0.5], 
                         [0.3229166666666667, 0.6953125, 0.5, 0.9765625], [0.5416666666666666, 0.6953125, 0.6458333333333334, 0.984375],
                         [0.052083333333333336, 0.5, 0.7604166666666666, 0.78125]]


    control_image = Image.open("./data/girl_depth.png")
    control_image = control_image.convert("RGB")
    # control_image = control_image.resize((args.height, args.width))

    image = pipe(
        prompt=img_prompt,
        prompt_2='$BREAKFLAG$'.join([img_prompt] + instance_prompts[0]),
        control_image=control_image,
        height=args.height,
        width=args.width,
        num_inference_steps=20,
        guidance_scale=10.0,
        generator=torch.Generator().manual_seed(seed),
        instance_box_list = instance_box_list,
        hard_control_steps = args.hard_control_steps,
        I2I_control_steps = args.i2i,
        I2T_control_steps = args.i2t,
        T2T_control_steps = args.t2t,
        T2I_control_steps = args.t2i,
        use_sam_enhance = args.use_sam_enhance,
        sam_predictor = sam_predictor if args.use_sam_enhance else None
        # instance_token_num = args.instance_token_num
    ).images[0]
    image.save("girl_rendering_sam_enhance.png")

