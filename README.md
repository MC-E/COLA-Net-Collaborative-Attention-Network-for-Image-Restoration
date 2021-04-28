# COLA-Net: Collaborative Attention Network for Image Restoration
This repository is for COLA-Net introduced in the following paper:
Chong Mou, [Jian Zhang](https://jianzhang.tech/), Xiaopeng Fan, Hangfan Liu, and Ronggang Wang, "COLA-Net: Collaborative Attention Network for Image Restoration", (IEEE Transactions on Multimedia 2021)

[\[Paper\]](https://ieeexplore.ieee.org/document/9369906/authors#authors) [\[arxiv\]](https://arxiv.org/abs/2103.05961)

[Our Team](http://villa.jianzhang.tech/)

## Requirements
The code is built based on [RNAN](https://github.com/yulunzhang/RNAN).
- Python 3.6
- PyTorch == 1.1.0
- numpy
- skimage
- cv2

The training datasets are available at [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and [SIDD](https://www.eecs.yorku.ca/~kamel/sidd/).
## Contents
1. [Introduction](#Introduction)
2. [Tasks](#Tasks)
3. [Citation](#Citation)
4. [Acknowledgements](#Acknowledgements)
## Introduction
In this paper we propose a model dubbed COLA-Net to exploit both local attention and non-local attention to restore image content in areas with complex textures and highly repetitive details, respectively. It is important to note that this combination is learnable and self-adaptive. To be concrete, for local attention operation, we apply local channel-wise attention on different scales to enlarge the size of receptive field of local operation, while for non-local attention operation, we develop a novel and robust patch-wise non-local attention model for constructing long-range dependence between image patches to restore every patch by aggregating useful information (self-similarity) from the whole image.

The pre-trained models are available at [Google Drive](https://drive.google.com/file/d/1FVfQuGMV-AzOEFZILkBcgZuABHs9pyd_/view?usp=sharing) and [PKU Drive](https://disk.pku.edu.cn:443/link/F7BC2A6DA955F1C89A03A0F9396C7DC2).
### Proposed COLA-Net
1. The gloabal architecture of our proposed COLA-Net.
![Network](/Figs/network.PNG)
2. The details of our proposed patch-based non-local attention method.
![Patch-based Non-local Method](/Figs/nl.PNG)
3. The visualization of the collaborative attention mechanism.
![Adaptive selection between local and non-local attention](/Figs/heat.PNG)
## Tasks
### Gray-scale Image Denoising 
![PSNR_DN_Gray](/Figs/PSNR_DN_Gray.PNG)
![Visual_DN_Gray](/Figs/Visual_DN_Gray.PNG)
### Real Image Denoising 
![PSNR_DN_Gray](/Figs/PSNR_DN_Real.PNG)
![Visual_DN_Gray](/Figs/Visual_DN_Real.PNG)
### Image Compression Artifact Reduction  
![PSNR_DN_Gray](/Figs/PSNR_CAR.PNG)
![Visual_DN_Gray](/Figs/Visual_CAR.PNG)
## Citation
If you find the code helpful in your resarch or work, please cite the following papers.
```
@inproceedings{zhang2019rnan,
    title={Residual Non-local Attention Networks for Image Restoration},
    author={Zhang, Yulun and Li, Kunpeng and Li, Kai and Zhong, Bineng and Fu, Yun},
    booktitle={ICLR},
    year={2019}
}

@article{mou2021cola,
  title={COLA-Net: Collaborative Attention Network for Image Restoration},
  author={Chong, Mou and Jian, Zhang and Xiaopeng, Fan and Hangfan, Liu and Ronggang, Wang},
  journal={IEEE Transactions on Multimedia},
  year={2021}
}
```
## Acknowledgements
This code is built on [RNAN (PyTorch)](https://github.com/yulunzhang/RNAN). We thank the authors for sharing their codes of RNAN.
