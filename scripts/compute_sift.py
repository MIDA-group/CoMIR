#@String(value='The path to all images of modality A.') pathA
#@String(value='The path to all images of modality B.') pathB
#@String(value='The csv file to write.') result

# This scripts register two lists of images together, pairwise.
# The algorithm used is SIFT, it is possible to change the parameters of the
# algorithm in the code below. The result is outputted in a CSV file named "result.txt".
# Script mostly modified from: https://www.ini.uzh.ch/~acardona/fiji-tutorial/#feature-extraction-sift-similarity
#
# Example:
# $fiji --ij2 --headless --run sift.py 'pathA="imageA.png",pathB="imageB.png",result="result.csv"'
#
# Notes:
# * It is possible to feed paths with wildcards in them, e.g.: /path/A/*.png
# * In the case of a list, images are sorted alphabetically.

# Python standard library
import glob
import sys
# ImageJ related
from ij import IJ
from mpicbg.imagefeatures import FloatArray2DSIFT, FloatArray2D
from mpicbg.models import PointMatch, RigidModel2D, NotEnoughDataPointsException
# Java imports
from java.lang import Double
from java.lang.reflect.Array import newInstance as newArray
from java.lang import System

# Loading the path of all images + sanity check
list1 = sorted(glob.glob(pathA))
list2 = sorted(glob.glob(pathB))
N = len(list1)
if len(list1) != len(list2):
    print("The list of image pairs must be of the same size.")
    print("Currently, there are %d images in set 1 and %d in set 2." % (
        len(list1), len(list2)
    ))
    System.exit(0)
else:
    print("%d IMAGE PAIRS ARE ABOUT TO BE PROCESSED!" % N)

# Resulting CSV file
# The results will be stored in a csv file
f = open(result, "w")
f.write("idx,fileA,fileB,m00,m10,m01,m11,m02,m12,cost\n")
print("The results will be stored in %s!" % result)
print

for i in range(N):
    print("+=============================================+")
    print("| Registration of image pairs {: 4d} / {: 4d}     |".format(i+1, N))
    print("+=============================================+")
    print

    # Loading the pair of images
    print("Loading the images...")
    imp1 = IJ.openImage(list1[i])
    print(imp1)
    imp2 = IJ.openImage(list2[i])
    print(imp2)
    print

    # Parameters for SIFT: NOTE 4 steps, larger maxOctaveSize
    p = FloatArray2DSIFT.Param()
    p.fdSize = 4 # number of samples per row and column
    p.fdBins = 8 # number of bins per local histogram
    p.maxOctaveSize = 1024 # largest scale octave in pixels
    p.minOctaveSize = 128   # smallest scale octave in pixels
    p.steps = 4 # number of steps per scale octave
    p.initialSigma = 1.6

    def extractFeatures(ip, params):
        sift = FloatArray2DSIFT(params)
        sift.init(FloatArray2D(ip.convertToFloat().getPixels(),
                                ip.getWidth(), ip.getHeight()))
        features = sift.run() # instances of mpicbg.imagefeatures.Feature
        return features

    print("Extracting SIFT features:")
    print("Image 1...")
    features1 = extractFeatures(imp1.getProcessor(), p)
    print("Image 2...")
    features2 = extractFeatures(imp2.getProcessor(), p)
    print

    # Closing the images and freeing the memory
    imp1.close()
    imp2.close()

    # Find matches between the two sets of features
    # (only by whether the properties of the features themselves match,
    #  not by their spatial location.)
    rod = 0.9 # ratio of distances in feature similarity space
              # (closest/next closest match)
    print("Matching features...")
    pointmatches = FloatArray2DSIFT.createMatches(features1, features2, rod)

    # Some matches are spatially incoherent: filter matches with RANSAC
    model = RigidModel2D() # supports translation, rotation and scaling
    candidates = pointmatches # possibly good matches as determined above
    inliers = [] # good point matches, to be filled in by model.filterRansac
    maxEpsilon = 25.0 # max allowed alignment error in pixels (a distance)
    minInlierRatio = 0.05 # ratio inliers/candidates
    minNumInliers = 5 # minimum number of good matches to accept the result

    try:
        modelFound = model.filterRansac(candidates, inliers, 1000,
                                        maxEpsilon, minInlierRatio, minNumInliers)
        if modelFound:
            # Apply the transformation defined by the model to the first point
            # of each pair (PointMatch) of points. That is, to the point from
            # the first image.
            PointMatch.apply(inliers, model)
    except NotEnoughDataPointsException, e:
        print e

    if modelFound:
        print("SUCCESS! Saving results...")
        # Register images
        rigid = newArray(Double.TYPE, (2, 3))
        model.toMatrix(rigid)
        line = "%d,%s,%s,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n" % (
            i, list1[i], list2[i],
            rigid[0][0], rigid[1][0],
            rigid[0][1], rigid[1][1],
            rigid[0][2], rigid[1][2],
            model.cost
        )
        f.write(line)
        print("Done.")
        print
    else:
        print("Some problem occured for image pair %d..." % (i+1))

f.close()
print("DONE.")
