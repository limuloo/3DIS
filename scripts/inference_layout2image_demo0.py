from diffusers import UniPCMultistepScheduler
from threeDIS.utils import seed_everything, load_migc
from threeDIS.pipeline_stable_diffusion_layout2depth import StableDiffusionL2DPipeline, MIGCProcessor
import os
import torch
from PIL import Image
from copy import deepcopy
import numpy as np
import argparse



parser = argparse.ArgumentParser(description='MIGC Inference')
# ../output/exp011/del_color_0_
# /home/zdw/project/output/exp011_131
parser.add_argument('--out_path', type=str, default='/data/hpfs/zdw/code/RD/3dis_v2/teaser4/output', help='output directory')
parser.add_argument('--depth_path', type=str, default='/data/hpfs/zdw/code/RD/3dis_v2/teaser4/depth', help='path to MIGC checkpoint')
parser.add_argument('--rgb_path', type=str, default='/data/hpfs/zdw/code/RD/3dis_v2/teaser4/depth', help='path to MIGC checkpoint')
# parser.add_argument('--instance_token_num', type=int, default=50)
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
    #######################  First Stage, Layout-to-Depth  #######################
    # Construct Layout-to-Depth pipeline
    sd_path = "Intel/ldm3d-4c" if not os.path.isdir('/mnt/sda/zdw/ckpt/ldm3d-4c') else '/mnt/sda/zdw/ckpt/ldm3d-4c'
    pipe = StableDiffusionL2DPipeline.from_pretrained(sd_path)
    
    # load fine-tuned Text-to-Depth model
    unet_path = './pretrained_weights/unet_0901.ckpt'
    assert os.path.isfile(unet_path), f"Checkpoint file not found at {unet_path}"
    pipe.unet.load_state_dict(torch.load(unet_path))

    # load Layout-to-Depth Adapter
    layout_adapter_ckpt_path = './pretrained_weights/layout_adapter.ckpt'
    assert os.path.isfile(layout_adapter_ckpt_path), f"Checkpoint file not found at {layout_adapter_ckpt_path}"
    load_migc(pipe.unet, layout_adapter_ckpt_path, attn_processor=MIGCProcessor)
    
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe = pipe.to("cuda")


    seed = 666666
    seed_everything(seed)
    

    img_prompt = "A wooden table holding multiple beers and a plate of food with pizza on it."
    instance_prompts = [["cup", 'cup', 'cup', 'dining table', 'pizza', 'pizza']]
    for instance_prompt in instance_prompts[0]:
        img_prompt += ',' + instance_prompt
    prompt_final = [[img_prompt]]
    for instance_prompt in instance_prompts[0]:
        prompt_final[0].append(instance_prompt)

    bboxes = [[[0.6449509803921568, 0.5280882352941176, 0.8853921568627451, 0.9258496732026144], 
               [0.459640522875817, 0.6456209150326797, 0.7286928104575164, 0.9797058823529412], 
               [0.7911274509803922, 0.4461764705882353, 1.0, 0.7921405228758169], 
               [0.0022549019607843133, 0.2, 0.9977450980392157, 0.9887581699346405], 
               [0.16405228758169935, 0.4157352941176471, 0.5820261437908497, 0.6561764705882354], 
               [0.471421568627451, 0.3274019607843137, 0.8109150326797385, 0.5453104575163399],
               ]]

    image = pipe(deepcopy(prompt_final), bboxes, num_inference_steps=30, guidance_scale=6.5, 
                                    MIGCsteps=15, NaiveFuserSteps=30, aug_phase_with_and=True,
                                    negative_prompt='worst quality, low quality, bad anatomy, worst quality, low quality, bad anatomy, watermark, text, blurry, extra hands').images[0]
    
    image_np = np.array(image)
    
    from PIL import Image
    image_rgb = Image.fromarray((image_np[..., 0: 3]).astype('uint8'))
    image = Image.fromarray((image_np[..., 3]).astype('uint8'))

    save_depth_path = "table_depth.png"
    image.save(save_depth_path)
    pipe = pipe.to('cpu')


    #######################  Sencond Stage, Rendering  #######################


    from threeDIS.pipeline_flux_rendering import FluxRenderingPipeline
    from threeDIS.detail_renderer.detail_renderer_flux import DetailRendererFLUX
    from threeDIS.utils import get_all_processor_keys

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
    
    control_image = Image.open(save_depth_path)
    control_image = control_image.convert("RGB")

    image = pipe(
        prompt=img_prompt,
        prompt_2='$BREAKFLAG$'.join([img_prompt] + instance_prompts[0]),
        control_image=control_image,
        height=args.height,
        width=args.width,
        num_inference_steps=20,
        guidance_scale=10.0,
        generator=torch.Generator().manual_seed(seed),
        instance_box_list = bboxes[0],
        hard_control_steps = args.hard_control_steps,
        I2I_control_steps = args.i2i,
        I2T_control_steps = args.i2t,
        T2T_control_steps = args.t2t,
        T2I_control_steps = args.t2i,
        use_sam_enhance = args.use_sam_enhance,
        sam_predictor = sam_predictor if args.use_sam_enhance else None
        # instance_token_num = args.instance_token_num
    ).images[0]
    image.save("table.png")
    