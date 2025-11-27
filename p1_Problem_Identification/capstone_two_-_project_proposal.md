# **Project Proposal: Image Processing for Glaucoma Detection**

## Problem Identification

### Problem Statement

1. Can a deep learning model trained on unimodal OCT data accurately classify glaucoma while maintaining performance across diverse patient subpopulations, including those with high myopia?  
2. (If time permits) Can integrating OCT, fundus imaging, and clinical metrics into a multimodal model improve the accuracy, robustness, and interpretability of glaucoma screening tools?

### Context

Glaucoma is a leading cause of irreversible blindness, affecting over 80 million people globally. Early detection is crucial to prevent vision loss, yet glaucoma often progresses silently until its later stages. Optical Coherence Tomography (OCT) is a key imaging tool for early diagnosis. However, machine learning models trained on unimodal OCT data frequently struggle to distinguish between glaucomatous changes and anatomical variations due to high myopia; while many recent models show high performance on general datasets, their ability to generalize across diverse patient populations remains limited.  Multimodal approaches integrating OCT with other imaging modalities (e.g., fundus photography) and tabular clinical data (e.g., intraocular pressure, age, refractive error) have shown improved performance. However, integration, interpretability, and scalability remain open challenges.

### Criteria for Success

* Reproduce a baseline OCT-based model with performance metrics comparable to published results, or potentially better performance with combined datasets.

* Evaluate model generalizability, especially in high-myopia subgroups.

* Create interpretable visualizations or model explanations.

* Deliver a complete GitHub repository, reproducible code, and a summary slide deck.

### Scope of Solution Space

The solution will focus on developing and evaluating deep learning models (e.g., CNNs, multimodal fusion models) for glaucoma classification.  The project will explore multiple datasets, emphasizing modularity and reproducibility. Potential model extensions may include feature-level fusion, attention-based architectures, or ensemble methods.

### Constraints

* Limited access to high-resolution labeled datasets with complete clinical metadata.

* GPU/compute limitations may constrain training on full-resolution volumes.

* Time constraints may limit the extent of hyperparameter tuning and advanced ensembling.

### Stakeholders

* **Primary Stakeholder**: Clinicians and healthcare researchers working in ophthalmology and medical imaging, who need accurate and interpretable diagnostic support tools.

* **Secondary Stakeholders**: Data scientists and developers creating medical ML tools; public health entities aiming to improve early glaucoma screening.

### Data Sources

#### *Unimodal*

* OCT Volumes for Glaucoma Detection (1110 OCT scans)

* Composite Retinal Fundus and OCT Dataset (with labeled glaucomatous disorders)

#### *Multimodal*

* PAPILA Dataset (includes demographic data and clinical metrics)

* GRAPE Dataset (multimodal records including visual field data, segmentations, etc.)

All datasets are publicly available and do not require special permissions.

---

## Project Outline

### Approach Overview

The project will be divided into two main phases:

**PHASE 1:**  A unimodal pipeline using OCT data only.  This will likely be framed as a supervised binary classification problem – predicting 1 or 0 for the presence or absence of glaucoma, respectively – but will ultimately depend on the nature of the datasets upon further inspection.

**PHASE 2:** If time permits, a multimodal extension of Phase 1 that integrates multiple imaging modalities, along with tabular clinical data (patient demographics, IOP measurements, etc.).  If possible, integration of time series analysis may be particularly helpful for some metrics – e.g., IOP & visual field trends, changes in OCT measurements over time.  This might also be framed as a supervised binary classification problem, but perhaps expansion of response categories or conversion to a regression problem (i.e., to provide gradations of risk for clinicians to incorporate into their final decision-making rather than black and white predictions) and/or partial incorporation of unsupervised learning (i.e., to address gaps between merged datasets) may be helpful.

Each phase will follow a standard data science pipeline to ensure modular, transparent, and reproducible development:

### Data Science Pipeline

#### *1\. Problem Identification* (discussed above)

#### *2\. Data Wrangling*

* Upload datasets into the appropriate programming environment (e.g., reading image arrays and metadata into data structures such as pandas DataFrames, NumPy arrays, open-cv or pillow image objects).  
* Standardize and transform inputs from different datasets to align on common formats (e.g., normalizing image dimensions, harmonizing label formats, aligning metadata fields).  
* Define and document variable names, data types, value ranges, and missing values.  
* Perform data cleaning, validation, and, where appropriate, imputation for incomplete records.

#### *3\. Exploratory Data Analysis (EDA)*

* Visualize and inspect sample OCT volumes and fundus images to understand data characteristics and potential quality issues.  
* Analyze distribution of labels (e.g., glaucoma vs. non-glaucoma) across datasets and stratify by potential confounders (e.g., refractive error, age).  
* Begin identifying patterns or associations between features and target labels.  
* At this stage, the ML framing will likely be clarified as a supervised binary classification problem \- i.e., since we will likely be relying on the labeling of data points with binary category for the presence or absence of glaucoma.  

#### *4\. Preprocessing and Training Data Development*

* Apply data augmentation or normalization strategies to prepare images for model training.  
* Encode categorical variables and scale numerical variables if working with tabular data.  
* Split the dataset into training, validation, and test sets using stratified sampling where applicable.

#### *5\. Modeling*

* Build a baseline convolutional neural network for unimodal OCT image classification.  
* Evaluate model performance and failure modes, especially in edge cases such as high-myopia samples.  
* Extend to a multimodal architecture by fusing additional data inputs, including fundus images and clinical metrics.  
* Explore use of attention mechanisms or late fusion techniques to improve multimodal interpretability and performance.

#### *6\. Documentation and Presentation*

* Maintain a GitHub repository with modular scripts or notebooks for each step of the pipeline.  
* Document methods, rationale, and dependencies clearly.  
* Produce a final slide deck and a written project report summarizing methodology, results, and key takeaways.

*\*Note that the above approach may be iterative, with flexibility for refinement at any step based on preliminary insights and results.*

---

## Deliverables

### Primary Deliverables

* GitHub Repository with:

  * Clean, modular code for data preprocessing, model training, evaluation, and visualization

  * Documentation and instructions for replicating all experiments

* Completed Jupyter Notebooks or Python modules for each development stage

* Final Slide Deck for presentation to stakeholders

* Written Project Report summarizing findings, methodology, and recommendations

### Intended Use

The project output will serve as both a diagnostic pipeline prototype and a demonstration of applying deep learning to real-world medical imaging challenges. It may support future development into a software tool, research paper, or internal benchmarking application for healthcare teams.

