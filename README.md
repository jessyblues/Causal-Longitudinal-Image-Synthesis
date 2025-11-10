# Causal-Longitudinal-Image-Synthesis
code for paper "CLIS: Causal Longitudinal Image Synthesis and its Application to Alzheimer’s Disease Characterization"


## 🧠 Introduction
This project provides a pytorch implementation of CLIS(Causal Longitudinal Image Synthesis), a causal model that overcomes these challenges through a novel integration of generative imaging, continuous-time modeling, and structural causal models combined with a neural network. 

Specifically, we first depict the causality between tabular variables including demographic variables, clinical biomarkers, and brain volume size via a tabular causal graph (TCG), and then further establish a tabular-visual causal graph (TVCG) to causally synthesize the brain MRI by developing an intervened MRI synthesis module as an bridge between TCG and MRI. It also introduces an independent variable to explicitly model the time interval. We train our CLIS based on the ADNI dataset and evaluate it on two other AD datasets to illustrate the outstanding yet controllable quality of the synthesized images and the contributions of synthesized MRI to AD characterization, substantiating its reliability and utility in clinics

## 📦 Installation
```bash
conda create -n clis python=3.10
pip install -r requirements.txt 

## Training StyleGAN
```bash
python -m StyleGAN.train_3D_style_GAN_DP

