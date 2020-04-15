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
    description="Converts pairs of multi-channel (e.g. RGB) images to grayscale."
)
parser.add_argument("pathA", type=str,
                   help="the path to the first set of images.")
parser.add_argument("pathB", type=str,
                   help="the path to the 2nd set of images.")
parser.add_argument("exportPath", type=str,
                   help="the path to export the images to.")
args = parser.parse_args()

# Creating the export folder
if not os.path.exists(args.exportPath):
    os.makedirs(args.exportPath)    
    print(f"{args.exportPath} was created")

# Finding the list of images to PCA down
list1 = sorted(glob.glob(args.pathA))
list2 = sorted(glob.glob(args.pathB))
N = len(list1)
if len(list1) != len(list2):
    print("The list of image pairs must be of the same size.")
    print("Currently, there are %d images in set 1 and %d in set 2." % (
        len(list1), len(list2)
    ))
else:
    print("%d IMAGE PAIRS ARE ABOUT TO BE PROCESSED!" % N)

# PROCESSING
pbar = tqdm(total=N)
for i in range(N):
    # Images naming
    image1_path = list1[i]
    image1_name = os.path.basename(image1_path)
    image1_export = os.path.join(
        args.exportPath,
        image1_name
    )
    image2_path = list2[i]
    image2_name = os.path.basename(image2_path)
    image2_export = os.path.join(
        args.exportPath,
        image2_name
    )
    pbar.set_description(image1_name)
    
    im1 = skio.imread(image1_path)
    im2 = skio.imread(image2_path)
    assert im1.shape == im2.shape, "The images must have the same size!"

    # Dimensionality reduction
    w, h, c = im1.shape
    pixels = np.vstack([
        im1.reshape(-1, c),
        im2.reshape(-1, c)
    ])
    result = PCA(n_components=1).fit_transform(pixels)
    
    final1 = result[:h*w].reshape(h, w)
    final2 = result[h*w:].reshape(h, w)
    
    skio.imsave(image1_export, final1)
    skio.imsave(image2_export, final2)
    
    pbar.update(1)

print("DONE.")