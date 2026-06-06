---
title: OCT Glaucoma Screening
emoji: 👁️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.8.0
app_file: app.py
pinned: false
license: mit
---

# OCT Glaucoma Screening Demo

**Feature-agnostic glaucoma detection from 3D OCT volumes using deep learning.**

Upload a preprocessed OCT volume (`.npy` file, shape `64×128×64`, uint8) and receive
a glaucoma screening prediction from one of three trained 3D CNN architectures:

| Model | Architecture | AUC | Parameters |
|-------|-------------|-----|-----------|
| Sequential CNN | 5× (Conv3D→BN→ReLU)→GAP→Dense | 0.88 | ~37K |
| ResNet-Like | Residual skip connections | 0.94 | ~89K |
| Attention Network | SE + spatial attention | 0.93 | ~112K |

## How to Use

1. Select a model architecture from the dropdown
2. Upload a `.npy` volume file OR select a sample volume
3. Explore slices using the axis selector and slider
4. Click "Run Screening" for the prediction

## Data Format

Volumes must be NumPy arrays with shape `(64, 128, 64)` and dtype `uint8`.
These are optic-nerve-head-centered OCT scans cropped and normalized to this standard size.

## Citation

Based on: Maetschke et al. (2019) "A feature agnostic approach for glaucoma detection
in OCT volumes" — [DOI: 10.1371/journal.pone.0219126](https://doi.org/10.1371/journal.pone.0219126)

Dataset: [Zenodo DOI: 10.5281/zenodo.1481223](https://zenodo.org/record/1481223)
