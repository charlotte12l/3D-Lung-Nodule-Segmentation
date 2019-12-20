# 3D Lung Nodule Segmentation
## Introduction

We proposed to use 2D (pretrained or not) ResUNet/Densesharp combined with [Temporal Shift Module](https://arxiv.org/abs/1811.08383)
to match the performance of 3D Densesharp in 3D Lung Nodule Segmentation tasks. We used LIDC Dataset.

Extensive ablation studies validated that this combination could improve the performance of 2D CNNs and even comparable to 3D CNNs.

## Files
1. Models
- model/densenet3d: Our [Densesharp](https://github.com/duducheng/DenseSharp) baseline
- model/densenet2d_tsm: The 2D version of Densesharp, combined with Temporal Shift Module(TSM)
- model/densenet2d_notsm: The 2D version of Densesharp, without TSM
- model/resnet18_2d_tsm: Resnet18UNet, combined with TSM
- model/resnet18_2d_notsm: Resnet18UNet, without TSM

2. Scripts
- scripts/train_densenet: train densenet
- scripts/train_resnet: train resnet
 
3. Utils
- utils/tsm.py: [Temporal Shift Module Implementation](https://github.com/PingchuanMa/Temporal-Shift-Module)

## Reference 
- [Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/abs/1811.08383)
- [3D Deep Learning from CT Scans Predicts Tumor Invasiveness of Subcentimeter Pulmonary Adenocarcinomas](https://cancerres.aacrjournals.org/content/78/24/6881.full)

## Acknowledgement
@duducheng ,@rongyaofang, @PingchuanMa
