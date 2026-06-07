# OCTCV — Glaucoma Detection from 3D OCT Volumes

> Deep learning for binary glaucoma classification using 3D Optical Coherence Tomography volume scans centered on the optic nerve head.

**Best Model: ResNet-Like | AUC = 0.94 | 114K params | 59s training**

---

## The Story

Glaucoma causes irreversible blindness in 80M+ people globally. Early detection matters, but distinguishing glaucomatous damage from normal anatomical variation on OCT scans is time-consuming for clinicians.

This project asks: *Can a 3D CNN learn to classify glaucoma from raw OCT volumes without manual feature extraction?*

**Part 1** reproduced the approach of [Maetschke et al. (2019)](https://doi.org/10.1371/journal.pone.0219126) and tested two additional architectures. The ResNet-Like model matched the paper's reported AUC of 0.94.

**Part 2** asked whether physics-informed data augmentation (6x expansion) could push performance further. The answer: **no** — augmentation does not substitute for patient diversity.

---

## Results at a Glance

| Architecture | Part 1 (888 vols) | Part 2 (5,328 vols) | Winner |
|:---|:---:|:---:|:---:|
| **Sequential** | 0.88 | 0.82–0.88 | Part 1 |
| **ResNet-Like** | **0.94** | 0.87–0.92 | Part 1 |
| **Attention** | 0.93 | 0.88–0.93 | Tie |

The performance ceiling of ~0.94 AUC is limited by dataset diversity (624 patients, single scanner, single institution) — not by training set volume or model capacity.

---

## Architectures

All models accept `(64, 128, 64, 1)` input — downsampled 3D ONH-centered OCT volumes.

```
Sequential:   5x (Conv3D -> BN -> ReLU) -> GAP -> Dense(1)
              Kernels: 7-5-5-3-3, 32 filters, ~323K params

ResNet-Like:  Conv3D stem -> 3x Residual Blocks -> GAP -> Dense(1)
              Skip connections for gradient flow, ~114K params

Attention:    ResNet + Squeeze-Excitation + Spatial Attention
              Channel & spatial recalibration, ~130K params
```

---

## Augmentation Pipeline (Part 2)

Five physics-informed transforms simulate realistic OCT imaging variation:

| Transform | Parameter | Simulates |
|:---|:---:|:---|
| Gamma Bright | γ = 1.67 | Over-exposed scan |
| Gamma Dark | γ = 0.60 | Under-exposed scan |
| Rayleigh Noise | scale = 0.67 | OCT speckle noise |
| Low-Pass Filter | r = 30 | Defocus / reduced resolution |
| Fan Distortion | pivot = 121 | Scan-head misalignment |

**Result**: 1,110 volumes → 6,660 volumes (6x). Each transform targets a different axis of real-world imaging variability, yet none improved downstream classification.

---

## Project Structure

```
OCTCV/
├── octcv/                          # Shared Python library
│   ├── mdl_lib/
│   │   ├── __init__.py             # XVolSet, train_model(), ModelEvaluator
│   │   ├── architectures.py        # buildSequential, buildResNet, buildAttnNN
│   │   └── callbacks.py            # LivePlot, EpochProgressBar
│   ├── arrViz.py                   # Volume visualization
│   └── system_monitoring.py        # GPU/RAM monitoring
│
├── p1–p7/                          # Part 1: Problem → Modeling → Report
│   └── p5_Modeling/modeling.ipynb   # Baseline training & evaluation
│
├── PART_2/                         # Part 2: Augmentation study
│   ├── DW-EDA.ipynb                # EDA: SSIM, power spectra, augmentation design
│   ├── PPs-Modeling.ipynb          # Preprocessing + Modeling pipeline
│   ├── model_metrics.json          # Full metrics & configuration
│   ├── Capstone-Three_Final_Report.pdf
│   └── Capstone-Three_Presentation.pptx
│
├── datasrc/                        # Data (not tracked — see Setup)
├── feature-agnostic-glaucoma-detection.pdf
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/chuotmd/OCTCV.git
cd OCTCV
conda create --name octcv python=3.12.11
conda activate octcv
pip install -r requirements.txt
```

**Data**: Download OCT volumes from [Zenodo](https://zenodo.org/records/1481223) (DOI: 10.5281/zenodo.1481223). Place `.npy` files in `datasrc/volumesOCT/`.

**GPU**: CUDA-capable GPU recommended. Tested on NVIDIA RTX 4070 Ti with TensorFlow 2.x.

---

## Training Configuration

```json
{
  "optimizer": "NAdam",
  "learning_rate": 1e-4,
  "loss": "binary_crossentropy",
  "batch_size": 4,
  "early_stopping": { "monitor": "val_auc", "patience": 5 },
  "input_normalization": "uint8 / 255 → [0, 1]"
}
```

---

## Key Takeaways

1. **Architecture > Data Volume** — ResNet's skip connections matter more than 6x more training data
2. **Augmentation ≠ New Patients** — Synthetic variation doesn't add anatomical diversity
3. **Small datasets can work** — 888 volumes is enough for 0.94 AUC with the right architecture
4. **The paper reproduces** — Our ResNet-Like matches the original authors' reported 0.94

---

## Data Sources

| Dataset | Use | Source |
|:---|:---|:---|
| OCT Volumes (Ishikawa 2018) | Primary — both parts | [Zenodo](https://zenodo.org/records/1481223) |
| Composite Fundus+OCT (Hassan 2021) | Exploratory — Part 1 | [Mendeley](https://data.mendeley.com/datasets/trghs22fpg/4) |

---

## References

1. Maetschke, S. et al. (2019). A feature agnostic approach for glaucoma detection in OCT volumes. *PLOS ONE*, 14(7), e0219126.
2. Ishikawa, H. (2018). OCT volumes for glaucoma detection. Zenodo. DOI: 10.5281/zenodo.1481223
3. He, K. et al. (2016). Deep residual learning for image recognition. *CVPR*.
4. Hu, J. et al. (2018). Squeeze-and-excitation networks. *CVPR*.

---

*Academic capstone project. OCT dataset available under its original Zenodo license.*
