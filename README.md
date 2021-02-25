# COLA-Net: Collaborative Attention Network for Image Restoration
This repository is for COLA-Net introduced in the following paper:
Chong Mou, Jian Zhang, Xiaopeng Fan, Hangfan Liu, and Ronggang Wang, "COLA-Net: Collaborative Attention Network for Image Restoration", (IEEE Transactions on Multimedia 2021)

The code is built on [RNAN](https://github.com/yulunzhang/RNAN).
## Contents
1. [Introduction](#Introduction)
2. [Tasks](#Tasks)
## Introduction
In this paper we propose a model dubbed COLA-Net to exploit both local attention and non-local attention to restore image content in areas with complex textures and highly repetitive details, respectively. It is important to note that this combination is learnable and self-adaptive. To be concrete, for local attention operation, we apply local channel-wise attention on different scales to enlarge the size of receptive field of local operation, while for non-local attention operation, we develop a novel and robust patch-wise non-local attention model for constructing long-range dependence between image patches to restore every patch by aggregating useful information (self-similarity) from the whole image.
### Proposed COLA-Net
![Network](/Figs/network.PNG)
![Patch-based Non-local Method](/Figs/nl.PNG)
![Adaptive selection between local and non-local attention](/Figs/heat.PNG)
## Tasks
### Gray-scale Image Denoising 
![PSNR_DN_Gray](/Figs/PSNR_DN_Gray.PNG)
![Visual_DN_Gray](/Figs/Visual_DN_Gray.PNG)
### Real Image Denoising 
![PSNR_DN_Gray](/Figs/PSNR_DN_Gray.PNG)
![Visual_DN_Gray](/Figs/Visual_DN_Gray.PNG)
### Image Compression Artifact Reduction  
![PSNR_DN_Gray](/Figs/PSNR_DN_Gray.PNG)
![Visual_DN_Gray](/Figs/Visual_DN_Gray.PNG)
