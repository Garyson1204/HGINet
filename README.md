# HGINet

Siyuan Yao, Hao Sun, Tian-Zhu Xiang, Xiao Wang and Xiaochun Cao

> The training and testing experiments are conducted using PyTorch with 4 NVIDIA GeForce RTX 3090 GPUs of 24 GB Memory.

------

## Prepare Data

The training and testing datasets can be downloaded at [COD10K-train](https://drive.google.com/file/d/1D9bf1KeeCJsxxri6d2qAC7z6O1X_fxpt/view), [COD10K-test + CAMO-test + CHAMELEON](https://drive.google.com/file/d/1QEGnP9O7HbN_2tH999O3HRIsErIVYalx/view) and [NC4K](https://drive.google.com/file/d/1kzpX_U3gbgO9MuwZIWTuRVpiB7V6yrAQ/view), respectively.

## Installation

```shell
conda create -n HGINet python=3.8
conda activate HGINet
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install openmim==0.3.9
mim install mmcv-full==1.7.0 mmengine==0.8.4
pip install mmsegmentation==0.30.0 timm h5py einops fairscale imageio fvcore pysodmetrics
```

## Training

```shell
python -m torch.distributed.launch --nproc_per_node=<#GPU> --master_port=<port> train.py --path "<COD dataset path>" --pretrain "<pretrain model path>"
```
Pretrain model weight is stored in [Google Drive](https://drive.google.com/file/d/1vdhUZ713peeo5hqXcdHPUdoMlhmDveu6/view?usp=drive_link). After downloading, please change the file path in the corresponding code or the training command.

## Testing & Evaluation

```shell
python test.py
python eval.py
```
Our well-trained model is stored in [Google Drive](https://drive.google.com/file/d/1bApr9EhHIdAzagLD_95aKBxwqv7reumX/view). After downloading, please change the file path in the corresponding code.

## Acknowledgement

This repository is built using the [BiFormer](https://github.com/rayleizhu/BiFormer) and [FSPNet](https://github.com/ZhouHuang23/FSPNet) repositories.
