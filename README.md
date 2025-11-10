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
  - Consistently named (e.g., `t1.nii.gz`)

---

## 🧠 Training StyleGAN

```bash
cd ./Causal-Longitudinal-Image-Synthesis
python -m StyleGAN.train_3D_style_GAN_DP -root_dir="./dataset"
```

## Training StyleGAN
```bash
cd ./Causal-Longitudinal-Image-Synthesis
python -m StyleGAN.train_3D_style_GAN_DP -root_dir="./dataset"
```





