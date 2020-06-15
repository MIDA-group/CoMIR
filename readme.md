[![License](https://img.shields.io/github/license/wahlby-lab/insilicotfm?style=flat-square)](https://opensource.org/licenses/MIT)
[![](https://img.shields.io/badge/python-3.6+-blue.svg?style=flat-square)](https://www.python.org/download/releases/3.6.0/) 

<p align="center">
  <img src="resources/comir_32.png" style="image-rendering: pixelated;"/>
</p>
<h1 align="center">CoMIR: <b>Co</b>ntrastive <b>M</b>ultimodal <b>I</b>mage <b>R</b>epresentation for Registration Framework</h1>
<h4 align="center">🖼 Registration of images in different modalities with Deep Learning 🤖</h4>

## Table of Contents

- [Introduction](#introduction)
- [Example](#example)
- [Quick Start Guide](#quick-start-guide)
- [Scripts](#scripts)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Introduction
This repository gives you access to the code necessary to:
* Train a Neural Network for converting images in a common latent space.
* Register images that were converted in the common latent space.

## Datasets

We use two datasets:
* Zurich Summer Dataset: https://sites.google.com/site/michelevolpiresearch/data/zurich-dataset
* Multimodal Biomedical Dataset for Evaluating Registration Methods: https://zenodo.org/record/3874362

## Reproduction of the results

All the results related to the Zurich sattelite images dataset can be reproduced
with the train-zurich.ipynb notebook. For reproducing the results linked to the
biomedical dataset follow the instructions below:

**Important:** for each script make sure you update the paths to load the correct
datasets and export the results in your favorite directory.

### Part 1. Training and testing the models
Run the notebook named train-biodata.ipynb. This repository contains a Release
which contains all our trained models. If you want to skip training, you can
fetch the models named model_biodata_mse.pt or model_biodata_cosine.pt and generate
the CoMIRs for the test set (last cell in the notebook).

### Part 2. Registration of the CoMIRs

Registration based on SIFT:
1. Compute the SIFT registration between CoMIRs (using Fiji v1.52p):
```bash
fiji --ij2 --run scripts/compute_sift.py 'pathA="/path/*_A.tif”,pathB="/path/*_B.tif”,result=“SIFTResults.csv"'
```
2. load the .csv file obtained by SIFT registration to Matlab
3. run evaluateSIFT.m

### Other results

Computing the registration with Mutual Information (using Matlab 2019b, use >2012a):
1. run RegMI.m
2. run Evaluation_RegMI.m

## Scripts
The script folder contains scripts useful for running the experiments, but also
notebooks for generating some of the figures appearing in the paper.

## Citation
Anonymized

## Acknowledgements
Anonymized
