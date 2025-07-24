## Overview

This project explores deep learning techniques for glaucoma detection using Optical Coherence Tomography (OCT) data. The primary goal is to investigate classification of optic nerve head (ONH)-centered 3D OCT volumes using convolutional neural networks (CNNs). A secondary goal is to explore whether 2D B-scan OCT images from a separate dataset can be incorporated into training or evaluation workflows, perhaps as a combined 2D image set (after extracting central / middle slices from the volumes of the first set).

The project is developed as part of a data science capstone and aims to:
- Reproduce and build upon prior research using public datasets
- Experiment with preprocessing pipelines for OCT image data
- Testing CNN architectures on OCT image classification tasks 
- Comparing performance under different data fusion strategies
- Documenting results for reproducibility and model interpretability (e.g., Class Activation Maps).

---

## Setup Instructions

This project is currently under development. The environment setup steps below are stable, but notebooks, models, and outputs are still being actively updated.

You can set up the environment as follows:

1. **Clone the repository**
```bash
git clone https://github.com/chuotmd/OCTCV.git
cd OCTCV
```
2. **Create and activate the environment / install dependencies**
```bash
conda create --name octcv python==3.12.11
conda activate octcv
```
3. **Install dependencies**
```bash
pip install -r requirements.txt
```
The primary code files will be jupyter notebooks using a python 3.12.11 kernel.

## Data Sources

### Dataset 1
**Name**: OCT volumes for glaucoma detection
OCT volumes for glaucoma detection
<br>**Contributors**: Ishikawa, Hiroshi
<br>**Affiliations**: New York University
<br>**Version**: 1.0.0
<br>**Published**: November 9, 2018
<br>**Source**: [Zenodo](https://zenodo.org/records/1481223) [[1](###references)]
<br>**DOI**: 10.5281/zenodo.1481223
<br>**Description**:
>"OCT scans centered on the ONH were acquired from 624 patients on a Cirrus SD-OCT Scanner (Zeiss, Dublin, CA, USA). The scans had physical dimensions of 6x6x2 mm with a corresponding size of 200x200x1024 voxels per volume. Scans with signal strength less than 7 were discarded, resulting in a total of 1110 scans for the experiments. The scans were kept in their original laterality (no flipping of left into right eye). 263 of the 1110 scans were diagnosed as healthy and 847 with primary open angle glaucoma (POAG). Glaucomatous eyes were defined as those with glaucomatous visual field defects (at least 2 consecutive abnormal test results)." [[2](###references)]

#### For This Project:
+ ***Location***: `./datasrc/volumesOCT/`
+ ***Notes***: 
  >At this time, using all 1110 Optic Nerve Head (ONH)-centered OCT volume scans, originally labeled as either <u>healthy</u> or <u>primary open angle glaucoma (POAG)</u> for binary classification.

### Dataset 2
**Name**: A Composite Retinal Fundus and OCT Dataset with Detailed Clinical Markings of Retinal Layers and Retinal Lesions to Grade Macular and Glaucomatous Disorders
<br>**Contributors**: Taimur Hassan, Muhammad Usman Akram, Muhammad Noman Nazir
<br>**Affiliations**: National University of Sciences and Technology, Khalifa University of Science and Technology
<br>**Version**: 4
<br>**Published**: September 22, 2021
<br>**Source**: [Mendeley](https://data.mendeley.com/datasets/trghs22fpg/4) [[3](###References)]
<br>**DOI**: 10.17632/trghs22fpg.4
<br>**Description**:
> "...composite retinal fundus and OCT dataset for analyzing retinal layers, retinal lesions, and to diagnose normal and abnormal retinal diseases like Centrally Involved Diabetic Macular Edema, Acute Central Serous Retinopathy (CSR), Chronic CSR, Geographic Age-related Macular Degeneration (AMD), Neovascular AMD, and Glaucoma." [[3](###References)],[[4](###References)],[[5](###References)]

#### For This Project:
+ ***Location***: `./datasrc/fundus-oct-composite/` 
+ ***Notes***: 
  >At this time, only using 2D B-SCAN OCT images originally labeled as either <u>glaucoma</u> or <u>normal</u>.

## References

[1] Ishikawa, H. (2018). *OCT volumes for glaucoma detection* (Version 1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.1481223

[2] Maetschke, S., Antony, B., Ishikawa, H., Wollstein, G., Schuman, J., & Garnavi, R. (2019). A feature agnostic approach for glaucoma detection in OCT volumes. *PLOS ONE*, 14(7), e0219126. https://doi.org/10.1371/journal.pone.0219126

[3] Hassan, T., Akram, M. U., & Nazir, M. N. (2021). *A Composite Retinal Fundus and OCT Dataset with Detailed Clinical Markings of Retinal Layers and Retinal Lesions to Grade Macular and Glaucomatous Disorders* (Version 4) [Data set]. Mendeley Data. https://doi.org/10.17632/trghs22fpg.4

[4] Hassan, T., Akram, M. U., Masood, M. F., & Yasin, U. (2018). Deep structure tensor graph search framework for automated extraction and characterization of retinal layers and fluid pathology in retinal SD-OCT scans. *Computers in Biology and Medicine*, 103, 58–68. https://doi.org/10.1016/j.compbiomed.2018.10.017

[5] Hassan, T., Akram, M. U., Werghi, N., & Nazir, M. N. (2020). RAG-FW: A hybrid convolutional framework for the automated extraction of retinal lesions and lesion-influenced grading of human retinal pathology. *IEEE Journal of Biomedical and Health Informatics*, 24(3), 792–802. https://doi.org/10.1109/JBHI.2019.2917361

