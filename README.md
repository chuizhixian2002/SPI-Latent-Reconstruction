# Physics-Guided Single-Pixel Image Reconstruction

This repository provides the PyTorch implementation of a physics-guided single-pixel imaging reconstruction framework.

The proposed framework contains two training stages:

1. Decoder pretraining from latent representations to image-domain reconstruction.
2. Guided SPI encoder training with a fixed decoder for latent-space and image-space supervision.

## Framework

The overall pipeline is:

```text
Input image
   ↓
SPI measurement layer
   ↓
Student encoder
   ↓
Estimated latent representation
   ↓
Fixed decoder
   ↓
Reconstructed image
