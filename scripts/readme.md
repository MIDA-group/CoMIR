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
The transformation is done by computing the principal components of each image per
pixel. The new images keep the same name and  are exported in the output folder. The
output folder is created if it does not exist. The images sets are sorted alphabetically.

**Requirements:** requirements.txt contains the needed python packages.

**Example:**
```bash
$ python3 scripts/nc2gray.py /path/*_A.png /path/*_B.png outputFolder
```
**Note:** The pair of images must have different names as they are exported in the same
folder.