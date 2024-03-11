# Official Pytorch Implementation of "Generation of 3D Brain MRI Using Auto-Encoding Generative Adversarial Networks" (accepted by MICCAI 2019)

This repository provides a PyTorch implementation of 3D brain Generation. It can successfully generates plausible 3-dimensional brain MRI with Generative Adversarial Networks. Trained models are also provided in this page.

## Paper
"Generation of 3D Brain MRI Using Auto-Encoding Generative Adversarial Networks"

The 22nd International Conference on Medical Image Computing and Computer Assisted Intervention(MICCAI 2019)
: (https://arxiv.org/abs/1908.02498)

## Cite
```
@inproceedings{kwon2019generation,
  title={Generation of 3D brain MRI using auto-encoding generative adversarial networks},
  author={Kwon, Gihyun and Han, Chihye and Kim, Dae-shik},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={118--126},
  year={2019},
  organization={Springer}
}
```

## Dependencies
* [Python 3.5+](https://www.continuum.io/downloads)
* [PyTorch 0.4.0+](http://pytorch.org/)
* [Jupyter Notebook](https://jupyter.org/)
* [Nilearn](https://nilearn.github.io/)
* [Nibabel](https://nipy.org/nibabel/)

We highly recommend you to use Jupyter Notebook for the better visualization!

## Dataset
You can download the Normal MRI data in [Alzheimer's Disease Neuroimaging Initiative(ADNI)](http://adni.loni.usc.edu/)
, Tumor MRI data in [BRATS2018](https://www.med.upenn.edu/sbia/brats2018/data.html) and Stroke MRI data in [Anatomical Tracings of Lesions After Stroke (ATLAS)](http://fcon_1000.projects.nitrc.org/indi/retro/atlas.html).

We converted all the DICOM(.dcm) files of ADNI into Nifti(.nii) file format using [SPM12](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/) I/O tools.

ADNI : Download Post-processed(processed with 'recon-all' command of [Freesurfer](https://surfer.nmr.mgh.harvard.edu/)) Structural images labeled as 'Control Normal'.

BRATS : Download dataset from BRATS2018 website.

ATLAS : Download dataset from ATLAS website.
        Obtain probability maps(masks) from the original .nii images with SPM12 'Segmentation' function. 
        Extract Brain areas with multiplying masks(c1,c2,c3 / GM,WM,CSF) with original images.

## Training Details
For each training, run 12,000 iterations (100 epochs in VAE-GAN)

Each run takes ~12 hour with one NVIDIA TITAN X GPU.

Run the Jupyter Notebook code for training (~train.ipynb)
        
## Test Details
You can download our Pre-trained models in our [Google Drive](https://drive.google.com/open?id=1Q5kkI_GxCY066c9owqzFFjzB_iEFCefJ)

Download the models and save them in the directory './checkpoint'
Then you can run the test code ('Test.ipynb')

Quantitative calculation (MS-SSIM / MMD score) & Image sampling is availble in the code.

For the PCA visualization, please follow the PCA tutorial that Nilearn provides.

## Model Details
You can get the detailed settings of used models in our model codes

(Model_alphaGAN.py , Model_alphaWGAN.py , Model_VAEGAN.py, Model_WGAN.py)


## Details for Dataset
If you have any question about data, feel free to e-mail me!

cyclomon@kaist.ac.kr
