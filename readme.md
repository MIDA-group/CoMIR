[![License](https://img.shields.io/github/license/wahlby-lab/insilicotfm?style=flat-square)](https://opensource.org/licenses/MIT)
[![](https://img.shields.io/badge/python-3.6+-blue.svg?style=flat-square)](https://www.python.org/download/releases/3.6.0/) 

<h1 align="center">Multimodal Image Registration Framework</h1>
<h4 align="center">🖼 Registration of images in different modalities with Deep Learning 🤖</h4>

## Table of Contents

- [Introduction](#introduction)
- [Example](#example)
- [Quick Start Guide](#quick-start-guide)
- [Scripts](#scripts)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Introduction
...

This repository gives you access to the code necessary to:
* Train a Neural Network for converting images in a common latent space.
* Register images that were converted in the common latent space.

## Example

## Quick Start Guide

## Scripts

### Compute SIFT

**Description:** Computes the SIFT features in two sets of images and finds the pairwise
transformations between both sets of images. The images sets are sorted alphabetically.
The results are exported in a csv file containing the rigid matrix transformation.

**Requirements:** Fiji must be installed.

**Example:**
```bash
$ fiji --ij2 --run scripts/compute_sift.py 'pathA="/path/*_A.png",pathB="/path/*_B.png",result="result.csv"'
```

### Convert N channels (e.g. RGB) images to grayscale (N-d to 1-d)

**Description:** Converts multichannel images, for instance RGB images, to grayscale.
The transformation is done by computing the principal components of a pair of images
and then reducing the dimension of both images. The new images keep the same name and
are exported in the output folder. The output folder is created if it does not exist.
The images sets are sorted alphabetically.

**Requirements:** requirements.txt contains the needed python packages.

**Example:**
```bash
$ python3 scripts/nc2gray.py /path/*_A.png /path/*_B.png outputFolder
```
**Note:** The pair of images must have different names as they are exported in the same
folder.

## Citation


## Acknowledgements