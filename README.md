# ZoRI: Towards Discriminative Zero-Shot Remote Sensing Instance Segmentation
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-%23EE4C2C.svg?style=&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python%203.8%20|%203.9-blue.svg?style=&logo=python&logoColor=ffdd54)](https://www.python.org/downloads/)

**[📄[arXiv]](https://arxiv.org/abs/2412.12798)**  &emsp; **[📄[PDF]](https://arxiv.org/pdf/2412.12798v1)** 

This repository contains code for our **AAAI2025** paper: [ZoRI: Towards Discriminative Zero-Shot Remote Sensing Instance Segmentation](https://arxiv.org/abs/2412.12798)  

<div align="center">
  <img src="imgs/framework.png" width="100%" height="100%"/>
</div><br/>

## Installation
Please see [Installation Instructions](INSTALL.md).

## Datasets
Download datasets for zero-shot remote sensing instance segmentation from One Drive.
###
[iSAID](https://entuedu-my.sharepoint.com/:u:/g/personal/shiqi006_e_ntu_edu_sg/EYMecJ1gGZ1FmrzPHLtAO-IBW2LZbgdw6WaiZufG41sfYA),
[NWPU](https://entuedu-my.sharepoint.com/:u:/g/personal/shiqi006_e_ntu_edu_sg/ERv0Jw_HYB1OvI-060K6BOcBabN5ej44l5x_fxPp-kofIw),
[SIOR](https://entuedu-my.sharepoint.com/:u:/g/personal/shiqi006_e_ntu_edu_sg/EZyiJWXVtsxAm3JsOs4eO-oBemergU9-Ef_27V8pJQz9yA?e=8gRJU6).

## Getting Started
Please see [Getting Started with ZoRI](GETTING_STARTED.md).

## Training
```
python train_net.py --config-file configs/zori_isaid_11_4.yaml
```
## Inference
###
For GZSRI setting, run
```
python train_net.py  --config-file configs/zori_isaid_11_4.yaml --eval-only MODEL.WEIGHTS [path_to_weights]
```
###
For ZSRI setting, run
```
python train_net.py  --config-file configs/zori_isaid_11_4.yaml --eval-only MODEL.WEIGHTS [path_to_weights] DATASETS.TEST '("isaid_zsi_11_4_val_unseen",)' MODEL.GENERALIZED False MODEL.CACHE_BANK.ALPHA 0.6
```
###
Then get pseudo unseen visual prototypes from previous inference results, run
```
python -m zori.utils.cache_model_unseen --config configs/cache.yaml DATASET 'isaid_zsi_11_4_val' PROTOTYPE_NUM [1] RESULTS [path_to_json]
```
###
Finally, inference again with pseudo unseen visual prototypes to get final predictions.


## Acknowledgement
This project is based on [FC-CLIP](https://github.com/bytedance/fc-clip). Many thanks to the authors for their great work!

## BibTeX
Please consider to cite ZoRI if it helps your research.

```bibtex
@inproceedings{ZoRI,
title={ZoRI: Towards Discriminative Zero-Shot Remote Sensing Instance Segmentation},
author={Huang, Shiqi and He, Shuting and Wen, Bihan},
booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
year={2025}
}
```

