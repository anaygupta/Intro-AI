# features.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# Anay Gupta
# CSE 471 - Project 4
# Prof Rao - T/TH 1:30 pm

# Resources referenced while working on project:
# https://en.m.wikipedia.org/wiki/Softmax_function
# https://www.geeksforgeeks.org/zip-in-python/
# https://www.pyimagesearch.com/2016/10/17/stochastic-gradient-descent-sgd-with-python/
# https://towardsdatascience.com/gradient-descent-in-python-a0d07285742f
# https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
# https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python


import numpy as np
import util
import samples
import matplotlib.pylab as plt # Yantian
from threading import Thread # Yantian
import time

DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28

def basicFeatureExtractor(datum):
    """
    Returns a binarized and flattened version of the image datum.

    Args:
        datum: 2-dimensional numpy.array representing a single image.

    Returns:
        A 1-dimensional numpy.array of features indicating whether each pixel
            in the provided datum is white (0) or gray/black (1).
    """
    features = np.zeros_like(datum, dtype=int)
    features[datum > 0] = 1
    return features.flatten()

def enhancedFeatureExtractor(datum):
    """
    Returns a feature vector of the image datum.

    Args:
        datum: 2-dimensional numpy.array representing a single image.
colorbar()
    Returns:
        A 1-dimensional numpy.array of features designed by you. The features
            can have any length.

    ## DESCRIBE YOUR ENHANCED FEATURES HERE...

    Implementing the feature suggested: finding the number of separate, connected regions of white pixels

    ##
    """
    features = basicFeatureExtractor(datum)

    "*** YOUR CODE HERE ***"
    # using identity of pixel (on or off) as feature
        # python dataClassifier.py --data largeMnist Dataset --feature_extractor basicFeatureExtractor --iter 1000 --learning_rate 0.1 --momentum 0.9

    # implementing the feature suggested: finding the number of separate, connected regions of white pixels
    rows = datum.shape[0]
    columns = datum.shape[1]
    visited = []
    nearby = []
    final = 0

    #iterate through 2d array (matrix of pixels)
    for i in range(rows):
        for j in range(columns):
            #declaring current pixel
            curr = datum[i][j]

            #check if current pixel is in visited list
            if (i,j) not in visited:
                #if not, add it to visited list (keeping tabs to avoid redundancy)
                visited.append((i,j))
                # check if current pixel is a white pixel
                if curr == 0:
                    # if current pixel is a white pixel, get all nearby neighbors in all four directions
                    temp = []
                    if (i < (rows - 1)):
                        temp.append((i + 1, j))
                    if (j < (columns - 1)):
                        temp.append((i, j + 1))
                    if (j > 0):
                        temp.append((i, j - 1))
                    if (i > 0):
                        temp.append((i - 1, j))
                    #add all of the elements in temp to the end of nearby list (neighbor tracking list)
                    nearby.extend(temp)
                    final = final + 1

                    # check neighbors that are also white pixels!
                    while len(nearby) != 0:
                        #pop first element from nearby list
                        points = nearby.pop(0)
                        #so long as it has not been visited
                        if points not in visited:
                            #get points of that pixel in the matrix
                            first = points[0]
                            second = points[1]
                            if datum[first][second] == 0:
                                temp = []
                                if (first < rows - 1):
                                    temp.append((first + 1, second))
                                if (second > 0):
                                    temp.append((first, second - 1))
                                if (first > 0):
                                    temp.append((first - 1, second))
                                if (second < columns - 1):
                                    temp.append((first, second + 1))
                                #add all of the nearby elements of the nearby elements to the end of the nearby list and continue until non-white pixel is reached
                                nearby.extend(temp)
                            #add it to visited list
                            visited.append((first, second))


    # one-hot encoding to be able to append it to feature matrix
    result = [0]*(8)
    result[final-1] = 1

    #append new feature to final features matrix
    newFeature = result
    features = np.append(features, newFeature)

    # util.raiseNotDefined()
    return features

def analysis(model, trainData, trainLabels, trainPredictions, valData, valLabels, validationPredictions):
    """
    This function is called after learning.
    Include any code that you want here to help you analyze your results.

    Use the print_digit(numpy array representing a training example) function
    to the digit

    An example of use has been given to you.

    - model is the trained model
    - trainData is a numpy array where each row is a training example
    - trainLabel is a list of training labels
    - trainPredictions is a list of training predictions
    - valData is a numpy array where each row is a validation example
    - valLabels is the list of validation labels
    - valPredictions is a list of validation predictions

    This code won't be evaluated. It is for your own optional use
    (and you can modify the signature if you want).
    """

    # Put any code here...
    # Example of use:
    # for i in range(len(trainPredictions)):
    #     prediction = trainPredictions[i]
    #     truth = trainLabels[i]
    #     if (prediction != truth):
    #         print "==================================="
    #         print "Mistake on example %d" % i
    #         print "Predicted %d; truth is %d" % (prediction, truth)
    #         print "Image: "
    #         print_digit(trainData[i,:])


## =====================
## You don't have to modify any code below.
## =====================

def print_features(features):
    str = ''
    width = DIGIT_DATUM_WIDTH
    height = DIGIT_DATUM_HEIGHT
    for i in range(width):
        for j in range(height):
            feature = i*height + j
            if feature in features:
                str += '#'
            else:
                str += ' '
        str += '\n'
    print(str)

def print_digit(pixels):
    width = DIGIT_DATUM_WIDTH
    height = DIGIT_DATUM_HEIGHT
    pixels = pixels[:width*height]
    image = pixels.reshape((width, height))
    datum = samples.Datum(samples.convertToTrinary(image),width,height)
    print(datum)

def _test():
    import datasets
    train_data = datasets.tinyMnistDataset()[0]
    for i, datum in enumerate(train_data):
        print_digit(datum)

if __name__ == "__main__":
    _test()
