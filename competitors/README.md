# Overview of the Competitor Methods

Here includes code used for the competitor methods and explanation of some implementation details. Please refer to their original repositories for their full code and documentation.

## Competitors

### pix2pix

- [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)
- [Project page](https://phillipi.github.io/pix2pix/)
- [Code used](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

### CycleGAN

- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
- [Project page](https://junyanz.github.io/CycleGAN/)
- [Code used](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

### DRIT++

- [DRIT++: Diverse Image-to-Image Translation via Disentangled Representations](https://arxiv.org/abs/1905.01270)
- [Project page](http://vllab.ucmerced.edu/hylee/DRIT_pp/)
- [Code used](https://github.com/HsinYingLee/DRIT)

## Modifications

### pix2pix

- `--preprocess pad`: Padding is added as an extra option for preprocessing during testing. This option will pad the input image sizes into minimum multiples of a certain divisor `d`. Function [`numpy.pad`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html) is used with `mode='reflect'`.
- `--divisor d`: Divisor `d` can be specified when using `--preprocess pad`. `d` is set to 256 by default. 

### CycleGAN

The same as `pix2pix`.

### DRIT++

- [test_transfer.py](./DRIT/src/test_transfer.py) – At test phase, instead of using the disentangled attribute representation of a random given image from the other modality, it searches the given image that is to be registered by filename.
  - For example, when translating `DATAROOT/testA/1.png` from modality `A` to `B`, it searches for `DATAROOT/testB/1.png` and uses the encoded attribute representation of `DATAROOT/testB/1.png`.
- For each `FILENAME`, the output images are `FILENAME_input.png`, `FILENAME_fake.png`, `FILENAME_real.png`.

## Environment

[environment.yml](./environment.yml) – Here includes the **full** list of packages used to run the experimentations. Some packages might be unnecessary.

## Code Overview

- `*/commands.sh` – These two files contain the commands with arguments that are used to run the full experiments.
- [utils/prepare_Eliceiri.py](./utils/prepare_Eliceiri.py) – This script is used to prepare the dataset for training. Whole slide images are augmented into patches. Augmentation details can be found at `class ImgAugTransform`.
- [utils/unpad_results.py](./utils/unpad_results.py) – This is a helper function to cut off the padded image borders during testing. 
    ```
    usage: unpad_results.py [-h] --path PATH [--width WIDTH] [--height HEIGHT]

    Unpad the result images to original sizes.

    optional arguments:
    -h, --help            show this help message and exit
    --path PATH, -p PATH  dir of images
    --width WIDTH         original width
    --height HEIGHT       original height
    ```




