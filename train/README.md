<a id="Train-Installation"></a>
## Training Environment Installation

### Conda environment setup
After setting up the [inference environment installation](../README.md#inference-installation), proceed to install the training environment.
```
conda activate 3DIS
cd train
pip install -r train_requirement.txt.txt
```


### Data Preparation
Download the processed VAE features ([000000_depth_latent_ldm3d](https://huggingface.co/limuloo1999/MIGC/tree/main/000000_depth_latent_ldm3d)) and annotations ([layout_anno_box_and_mask_v4](https://huggingface.co/limuloo1999/MIGC/tree/main/layout_anno_box_and_mask_v4)), and organize them as follows:

```
3DIS
├── pretrained_weights
│   ├── unet_0901.ckpt
│   ├── ...
├── threeDIS
│   ├── ...
├── scripts
│   ├── ...
├── train
│   └── dataset
│       ├── 000000_depth_latent_ldm3d
│       │   ├── data.mdb
│       │   └── lock.mdb
│       └── layout_anno_box_and_mask_v4
│           └── 000000.json
```


### Begin Training
You can quickly start training using the script below. Feel free to modify the parameters in `../configs/train.yaml` as needed.

```
cd train/scripts
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0  --master_port=10046 train_adapter_on_depth.py --yml_path ../configs/train.yaml --output_dir=../outputs_3dis_train
```