# The German Traffic Sign Recognition Benchmark
#
# sample code for reading the traffic sign images and the
# corresponding labels
#
# example:
#            
# trainImages, trainLabels = readTrafficSigns('GTSRB/Training')
# print len(trainLabels), len(trainImages)
# plt.imshow(trainImages[42])
# plt.show()
#
# have fun, Christian

import matplotlib.pyplot as plt
import csv
from pathlib import PurePath

ROOTPATH_TESTING = PurePath("../data/GTSRB", "testing", "images")
ROOTPATH_META = PurePath("../data/GTSRB", "META")
ROOTPATH_TRAINING = PurePath("../data/GTSRB", "training", "images")

def readTrainingTrafficSigns(rootpath):

    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    for c in range(0,43):
        prefix = PurePath(rootpath, f"{c:05d}") # subdirectory for class
        gtFile = open(PurePath(prefix, f"GT-{c:05d}.csv")) # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader) # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            image = plt.imread(str(PurePath(rootpath, row[0]))) # the 1th column is the filename
            image_width = row[1]
            image_height = row[2]
            image_roi_x1 = row[3]
            image_roi_y1 = row[4]
            image_roi_x2 = row[5]
            image_roi_y2 = row[6]
            image_label = row[7]
            images.append(image) # the 1th column is the filename
            labels.append(image_label) # the 8th column is the label
        gtFile.close()
    return images, labels



def readTestingTrafficSigns(rootpath):
    images = [] # images
    
    gtFile = open(PurePath(rootpath, f"GT-final_test.test.csv")) # annotations file
    gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
    next(gtReader) # skip header
    
    for row in gtReader:
        image = plt.imread(str(PurePath(rootpath, row[0]))) # the 1th column is the filename
        image_width = row[1]
        image_height = row[2]
        image_roi_x1 = row[3]
        image_roi_y1 = row[4]
        image_roi_x2 = row[5]
        image_roi_y2 = row[6]
        images.append(image) # the 1th column is the filename
    gtFile.close()
    return images

def readMetaTrafficSigns(rootpath):
    
    images = [] # images
    
    for c in range(0,43):
        image = plt.imread(str(PurePath(rootpath, f"{c}.png"))) # the 1th column is the filename
        plt.imshow(image)
        plt.show()
        images.append(image) # the 1th column is the filename
    return images

# trainImages, trainLabels = readTrainingTrafficSigns(ROOTPATH_TRAINING)
# testImages = readTestingTrafficSigns(ROOTPATH_TESTING)
metaImages = readMetaTrafficSigns(ROOTPATH_META)