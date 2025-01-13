def show_mask(mask, img):
    # print(f"img shape: {img.shape}") # （512, 512, 3）
    # print(f"mask shape: {mask.shape}") # (512, 512)
    random_color = np.random.randint(0, 255, size=3)
    alpha = 0.5
    color_mask = np.zeros_like(img)
    for i in range(3):
        color_mask[:, :, i] = random_color[i]
    mask = mask * 255
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask = mask.astype(img.dtype)
    img = cv2.addWeighted(img, alpha, color_mask, 1 - alpha, 0)
    img = cv2.addWeighted(img, 1, mask, 0.5, 0)
    return img
    
import torch
import os
import numpy as np
import cv2
from PIL import Image

from tqdm import tqdm
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2

sam2_checkpoint = "../../ckpt/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

predictor = build_sam2(model_cfg, sam2_checkpoint, device='cuda')
predictor = SAM2ImagePredictor(predictor)

annotation_path = '../MIGC/bench_file/mig_bench_anno.yaml'

import yaml
with open(annotation_path, 'r') as f:
    cfg = f.read()
    annatation_data = yaml.load(cfg, Loader=yaml.FullLoader)

bench_file_path = '../MIGC/bench_file/mig_bench.txt'
image_path = '../MIGC/exp011_131'
bench_name = os.path.split(bench_file_path)[-1][:-4]
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    with open(bench_file_path, 'r') as f:
        for prompt_line in tqdm(f):
            prompt = prompt_line.split('\n')[0]
            img_name = prompt + '_420_'
            coco_id = 0
            bboxes = []
            for phase in annatation_data[prompt]:
                if phase == 'coco_id':
                    coco_id = annatation_data[prompt][phase]
                    continue
                bbox_list = annatation_data[prompt][phase]
                for bbox in bbox_list:
                    bbox = [i * 512 for i in bbox]
                    bboxes.append(bbox)
                    
            # print(bboxes)

              
            img = cv2.imread(os.path.join(image_path, f"{img_name}{coco_id}.png"))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # print(os.path.join(image_path, f"{img_name}{coco_id}.png"))
            img_pil = Image.fromarray(img)
            predictor.set_image(img)
            input_box = np.array(bboxes)
            
            masks, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
            
            print(masks[0])
            
            # print(masks.shape)
            
            for mask in masks:
                img = show_mask(mask[0], img)
                
            if not os.path.exists('result'):
                os.makedirs('result')
                
            cv2.imwrite(f'result/masks_{img_name}{coco_id}.png', img)
            