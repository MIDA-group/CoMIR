# -*- coding: utf-8 -*-
# Python Standard Libraries
from glob import glob
import os, argparse

# ~ Scientific
import numpy as np
# ~ Image manipulation / visualisation
import matplotlib.pyplot as plt
import skimage.io as skio
# ~ Other
from tqdm import tqdm

# %%

def unpad_results(res_dir, wo, ho):
#    res_dir = './pytorch-CycleGAN-and-pix2pix/results/eliceiri_p2p_rotation_a2b/test_latest/images'
#    res_dir = './pytorch-CycleGAN-and-pix2pix/results/eliceiri_p2p_rotation_a2b'
#    (wo, ho) = (834, 834)
    def unpad_sample(img, wo, ho):
    #    img = skio.imread('./pytorch-CycleGAN-and-pix2pix/results/eliceiri_p2p_rotation_a2b/1B_A1_R_real_A.png')
    #    (wo, ho) = (834, 834)
        
        (wi, hi) = img.shape[:2]
        assert wo <= wi and ho <= hi
        wl = (wi - wo) // 2
        hl = (hi - ho) // 2
        return img[wl:wl+wo, hl:hl+ho]

    img_dirs = glob(f'{res_dir}/*')
    for img_path in tqdm(img_dirs):
        img = skio.imread(img_path)
        img = unpad_sample(img, wo, ho)
        skio.imsave(img_path, img)
    return

# %%
parser = argparse.ArgumentParser(description='Unpad the result images to original sizes.')
parser.add_argument(
        '--path', '-p', 
        required=True,
        help="dir of images")
parser.add_argument(
        '--width', 
        help="original width", 
        type=int, 
        default=834)
parser.add_argument(
        '--height', 
        help="original height", 
        type=int, 
        default=834)
args = parser.parse_args()

unpad_results(args.path, args.width, args.height)
#unpad_results('./pytorch-CycleGAN-and-pix2pix/results/eliceiri_p2p_rotation_b2a/test_latest/images', 834, 834)

