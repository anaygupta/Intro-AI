# answers.py
# ----------
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

import util


def q2():
    "*** YOUR CODE HERE ***"
    return ''

def q3():
    "*** YOUR CODE HERE ***"
    return ''
def q7():
    "*** YOUR CODE HERE ***"
    #Instead of using the perceptron algorithm, we are now minimizing a loss function with optimization techniques. Does this approach improve convergence?
    #visualize the Softmax Regression's training process on two 2D data sets by running the following commands:
        #python dataClassifier.py --model SoftmaxRegressionModel --data datasetA --iter [iterations]
        #python dataClassifier.py --model SoftmaxRegressionModel --data datasetB --iter [iterations]
    #when both datasets were tested starting from 50 to 200, both converged (?)
    return 'both'

def q10():
    """
    Returns a dict of hyperparameters.

    Returns:
        A dict with the learning rate and momentum.

    You should find the hyperparameters by empirically finding the values that
    give you the best validation accuracy when the model is optimized for 1000
    iterations. You should achieve at least a 97% accuracy on the MNIST test set.
    """
    hyperparams = dict()
    "*** YOUR CODE HERE ***"
    # filter out any item in the dict that is not the learning rate nor momentum
    allowed_hyperparams = ['learning_rate', 'momentum']
    hyperparams = {allowed_hyperparams[0]:.1, allowed_hyperparams[1]:.5}
    hyperparams = dict([(k, v) for (k, v) in hyperparams.items() if k in allowed_hyperparams])
    return hyperparams
