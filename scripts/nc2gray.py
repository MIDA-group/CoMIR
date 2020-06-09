# Python imports
import argparse
import glob
import os
# Other imports
import numpy as np
import skimage.io as skio
from sklearn.decomposition import PCA
from tqdm import tqdm

# Command line arguments
parser = argparse.ArgumentParser(
    description="Converts multi-channel (e.g. RGB) images to grayscale."
)
parser.add_argument("path", type=str,
                   help="the path to the set of images to convert.")
parser.add_argument("exportPath", type=str,
                   help="the path to export the images to.")
args = parser.parse_args()

# Creating the export folder
if not os.path.exists(args.exportPath):
    os.makedirs(args.exportPath)    
    print(f"{args.exportPath} was created")

# Finding the list of images to PCA down
imlist = sorted(glob.glob(args.path))
N = len(imlist)
print("%d IMAGES ARE ABOUT TO BE PROCESSED!" % N)

# PROCESSING
pbar = tqdm(total=N)
for i in range(N):
    # Images naming
    image_path = imlist[i]
    image_name = os.path.basename(image_path)
    image_export = os.path.join(
        args.exportPath,
        image_name
    )
    pbar.set_description(image_name)
    
    im = skio.imread(image_path)

    # Dimensionality reduction
    w, h, c = im.shape
    pixels = im.reshape(-1, c)
    result = PCA(n_components=1).fit_transform(pixels)
    
    final = result.reshape(h, w)
    
    skio.imsave(image_export, final)

    pbar.update(1)

print("DONE.")