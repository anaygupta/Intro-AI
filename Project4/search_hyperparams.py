# search_hyperparams.py
# ---------------------
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
import solvers
import util


def search_hyperparams(train_data, train_labels, val_data, val_labels,
                       learning_rates, momentums, batch_sizes, iterations,
                       model_class, init_param_values=None, use_bn=False):
    """
    Question 8: Evaluate various setups of hyperparameter and find the best one.

    Args:
        learning rate, momentums, batch_sizes are lists of the same length.
        The N-th elements from the lists form the N-th hyperparameter tuple.

    Returns:
        A model that corresponds to the best hyperparameter tuple, and the index
            of the best hyperparameter tuple

    Your implementation will train a model using each hyperparameter tuple and
    compares their accuracy on validation set to pick the best one.

    You must use MinibatchStochasticGradientDescentSolver.

    Useful methods:
    solver.solve(...)
    model.accuracy(...)
    """
    # Check length of inputs all the same
    hyperparams = [learning_rates, momentums, batch_sizes]
    for hyperparam in hyperparams:
        if len(hyperparam) != len(hyperparams[0]):
            raise ValueError('The hyperparameter lists need to be equal in length')
    hyperparams = zip(*hyperparams)
    # python *+VARIABLE: http://stackoverflow.com/questions/11315010/what-do-and-before-a-variable-name-mean-in-a-function-signature

    # Initialize the models
    models = []
    for learning_rate, momentum, batch_size in hyperparams:
        try:
            model = model_class(use_batchnorm=use_bn)
        except:
            model = model_class()
        if init_param_values is None:
            init_param_values = model.get_param_values()
        else:
            model.set_param_values(init_param_values)
        models.append(model)

    val_accuracies = []

    #declare and instantiate bestAcc once before as opposed to using -99999
    solver = solvers.MinibatchStochasticGradientDescentSolver(learning_rate, iterations, batch_size, momentum)
    solver.solve(train_data, train_labels, val_data, val_labels, models[0])
    bestAcc = model.accuracy(val_data, val_labels)

    # Loop over hyperparams
    for model, (learning_rate, momentum, batch_size) in zip(models, hyperparams):
        "*** YOUR CODE HERE ***"

        #all given parameters:
            #train_data, train_labels, val_data, val_labels,learning_rates, momentums, batch_sizes, iterations, model_class

        #from a given set of values for the learning rate, momentum and batch size, return the combination of values that results in highest validation set accuracy

        #you must use the minibatch SGD solver MinibatchStochasticGradientDescentSolver
        solver = solvers.MinibatchStochasticGradientDescentSolver(learning_rate, iterations, batch_size, momentum)
        #train on training set, evaluate on validation set

        #using solver.solve to train
        #calls def solve(self, input_train_data, target_train_data, input_val_data, target_val_data, model, callback=None):
        solver.solve(train_data, train_labels, val_data, val_labels, model)

        #use model.accuracy to evaluate
        acc = model.accuracy(val_data, val_labels)
        if (acc > bestAcc):
            bestAcc = acc
            bestHyperparams = [learning_rate, momentum, batch_size]
            bestModel = model

        # each accuracy is appended in order to the array (consistent with index of model and hyperparams)
        # val_accuracies.append(acc)

    # #set maxAcc to max accuracy
    # maxAcc = max(val_accuracies)
    # #get index of max accuracy
    # maxIndex = argmax(val_accuracies)
    # #get corresponding best model and hyperparams
    # bestModel = models[maxIndex]

    return bestModel, bestHyperparams

    # util.raiseNotDefined()
