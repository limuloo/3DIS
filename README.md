# 3DIS: DEPTH-DRIVEN DECOUPLED INSTANCE SYN- THESIS FOR TEXT-TO-IMAGE GENERATION


[[Project Page]](https://limuloo.github.io/3DIS/)  [[3DIS Paper]](https://arxiv.org/pdf/2410.12669) [[3DIS-FLUX Paper]](https://arxiv.org/pdf/2501.05131)


![attr_control](fig1.png)
## To Do List
- [x] Code
- [x] pretrained weights
- [ ] More demos

<a id="Installation"></a>
## Installation

### Conda environment setup
```
conda create -n 3DIS python=3.10 -y
conda activate 3DIS
pip install -r requirement.txt
pip install -e .
cd segment-anything-2
pip install -e . --no-deps
cd ..
```

### Checkpoints 🚀
**Step1** Download the checkpoint of the fine-tuned Text-to-Depth model, [unet_0901.ckpt](https://drive.google.com/file/d/1JJt5HPtVravufxSz46x__0ASt7b6fYel/view?usp=sharing).

**Step2** Download the checkpoint of the trained Layout-to-Depth Adapter, [layout_adapter.ckpt](https://drive.google.com/file/d/19rHJYrafOCQQk-zFj692vCYVNcLm1b9g/view?usp=sharing).

**Step3** Download the checkpoint of the SAM2, [sam2_hiera_large.pt](https://drive.google.com/file/d/1QjdY64w7pKm8smh0bV7K9-joeZiow8e0/view?usp=sharing).

**Step4** put them under the 'pretrained_weights' folder.
```
├── pretrained_weights
│   ├── unet_0901.ckpt
│   ├── layout_adapter.ckpt
│   ├── sam2_hiera_large.pt
├── threeDIS
│   ├── ...
├── scripts
│   ├── ...
```

## Layout-to-Depth Generation 🎨
### Single Image Example
You can quickly run inference for layout-to-depth generation using the following command:
```
python scripts/inference_layout2depth_demo0.py
```
<p align="center">
  <img src="figures/layout2depth_demo0.png" alt="example" width="500" height="250"/>
</p>

## Rendering with Various Models 🌈
### Rendering with FLUX
You can quickly run inference for FLUX rendering using the following command:
```
python scripts/inference_flux_rendering_sam_demo0.py  --width=768 --height=1024 --i2i=4 --use_sam_enhance
```
<p align="center">
  <img src="figures/flux_rendering_demo0.png" alt="example" width="800" height="400"/>
</p>

**More interesting demos will be coming soon!!!**


## Citation
If you find this repository useful, please use the following BibTeX entry for citation.
```
@article{zhou20243dis,
  title={3dis: Depth-driven decoupled instance synthesis for text-to-image generation},
  author={Zhou, Dewei and Xie, Ji and Yang, Zongxin and Yang, Yi},
  journal={arXiv preprint arXiv:2410.12669},
  year={2024}
}

@article{zhou20253disflux,
  title={3DIS-FLUX: simple and efficient multi-instance generation with DiT rendering},
  author={Zhou, Dewei and Xie, Ji and Yang, Zongxin and Yang, Yi},
  journal={arXiv preprint arXiv:2501.05131},
  year={2025}
}
```