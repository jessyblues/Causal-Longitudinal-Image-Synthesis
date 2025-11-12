# Causal-Longitudinal-Image-Synthesis
code for paper "CLIS: Causal Longitudinal Image Synthesis and its Application to Alzheimer’s Disease Characterization"


## 🧠 Introduction
This project provides a pytorch implementation of CLIS(Causal Longitudinal Image Synthesis), a causal model that overcomes these challenges through a novel integration of generative imaging, continuous-time modeling, and structural causal models combined with a neural network. 

Specifically, we first depict the causality between tabular variables including demographic variables, clinical biomarkers, and brain volume size via a tabular causal graph (TCG), and then further establish a tabular-visual causal graph (TVCG) to causally synthesize the brain MRI by developing an intervened MRI synthesis module as an bridge between TCG and MRI. It also introduces an independent variable to explicitly model the time interval. We train our CLIS based on the ADNI dataset and evaluate it on two other AD datasets to illustrate the outstanding yet controllable quality of the synthesized images and the contributions of synthesized MRI to AD characterization, substantiating its reliability and utility in clinics

## 📦 Installation
```bash
conda create -n clis python=3.10
pip install -r requirements.txt
```

## 📁 Dataset Structure

The training script expects the dataset directory (`./dataset`) to follow the structure below:

```
dataset/
├── ADNI/
│   ├── SUBJECT_001/
│   │   ├── 2005-09-18/
│   │   │   └── t1.nii.gz
│   │   ├── 2007-08-21/
│   │   │   └── t1.nii.gz
│   │   └── ...
│   └── ...
├── NACC/
│   ├── SUBJECT_002/
│   │   ├── 2014-03-10/
│   │   │   └── t1.nii.gz
│   │   ├── 2016-07-22/
│   │   │   └── t1.nii.gz
│   │   └── ...
│   └── ...
```

### ✅ Requirements

- The directory `./dataset` should contain multiple dataset sources such as `ADNI/`, `NACC/`, etc.
- Each source directory contains one folder per subject (e.g., `SUBJECT_001/`).
- Each subject folder contains multiple subfolders named by acquisition date (e.g., `2005-09-18/`).
- Each date folder must include **one preprocessed T1 MRI file** (`.nii.gz`), which should be:
  - Skull-stripped
  - Aligned to MNI space
  - Resampled to a uniform resolution

---

## Training StyleGAN
```bash
cd ./Causal-Longitudinal-Image-Synthesis
python -m StyleGAN.train_3D_style_GAN_DP --root_dir="./dataset"
```

## Training W encoder
```bash
python -m ImageEncoder.scripts.train
python -m ImageEncoder.training.train_noise_encoder
```

## Infer w and noise
```bash
python -m ImageEncoder.scripts.inference_encoder_for_GAN \
  --gan_pth=/your/path/of/trained/gan \
  --encoder_pth=/your/path/of/trained/encoder \
  --img_dir=/your/path/of/images
```

## ISM establishment


## Data Preprocessing for Causal Discovery and SCM Fitting

This script prepares the longitudinal dataset used for causal discovery and causal structural model (SCM) fitting.  
It performs the following steps:

1. Read raw data from ./dataset/raw_data.csv
2. Clean and normalize continuous variables
3. Construct longitudinal samples by pairing consecutive visits (T0 → T1) for each subject
4. Save processed files:
   - ./dataset/processed_data.csv — longitudinal normalized data
   - ./dataset/mean_and_std.csv — mean and standard deviation for normalization

---

### Input CSV format

The raw data file should include the following columns:

PTAU, ABETA42, TAU, SegVentricles, WholeBrain, GreyMatter, PTEDUCAT, Sex, APOE4, Age, Subject

Each row corresponds to one subject at one visit.  
The script will automatically group by Subject and construct paired longitudinal samples based on ascending Age.

---

### Normalization Rules

All variables except for Sex, APOE4, and PTEDUCAT are normalized using the mean and standard deviation computed from the raw data.

Variable Group | Variables | Normalized
---------------|-----------|----------
Biomarkers | PTAU, ABETA42, TAU | Yes
Demographics | Sex, Age, PTEDUCAT, APOE4 | Only Age and PTEDUCAT normalized
Brain Volumes | SegVentricles, WholeBrain, GreyMatter | Yes

The normalization statistics are saved to mean_and_std.csv.

---

### Output CSV format

The processed longitudinal dataset has the following column structure:

PTAU_T0, PTAU_T1, ABETA42_T0, ABETA42_T1, TAU_T0, TAU_T1,
SegVentricles_T0, SegVentricles_T1, WholeBrain_T0, WholeBrain_T1,
GreyMatter_T0, GreyMatter_T1, PTEDUCAT, Sex, APOE4, Age_T0, Age_T1, Subject

---

## Run the Causal Discovery and SCM Fitting script
```bash
python -m causal_discovery_and_SCM_fitting.process_data
python -m causal_discovery.causal_discovery --data_csv="./dataset/processed_data.csv" --output_dir="./causal_discovery"
python -m causal_discovery.causal_mlp --causal_graph_path="./causal_discovery/final_ensemble_graph.osm" --data_csv="./dataset/processed_data.csv" --output_dir="./causal_discovery"
```

