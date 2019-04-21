from __future__ import print_function

from past.builtins import xrange
import os
import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt


'''
Linear Support Vector Classification.

Similar to SVC with parameter kernel=’linear’, but implemented in terms of liblinear rather than libsvm, so it has more flexibility in the choice of penalties and loss functions and should scale better to large numbers of samples.
'''


# noinspection PyPackageRequirements
def svm_loss_naive(W, X, y, reg):
  """

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Structured SVM loss function, naive implementation (with loops).
  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0

  countIncorrects = 0 #counts scores out of the margin

  for i in xrange(num_train):
    scores = X[i].dot(W) # X[i] = 1,3073  W = 3073,10
    correct_class_score = scores[y[i]]

    countIncorrects = 0

    for j in xrange(num_classes):
      if j == y[i]:
        continue

      margin = scores[j] - correct_class_score + 1 # note delta = 1

      if margin > 0:
        loss += margin
        countIncorrects +=1

        # gradient update for incorrect rows
        dW[:,j] += 1 * X[i]

    # gradient update for the correct rows
    dW[:,y[i]] += -1 * countIncorrects * X[i]

  #original code
  # for i in xrange(num_train):
  #   scores = X[i].dot(W)
  #   correct_class_score = scores[y[i]]
  #   for j in xrange(num_classes):
  #     if j == y[i]:
  #       continue
  #     margin = scores[j] - correct_class_score + 1 # note delta = 1
  #     if margin > 0:
  #       loss += margin


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  # adding regularization to the gradient
  dW += reg * 2 * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  ## Calculate scores
  # calculates dot product of input with weights
  scores = X.dot(W) # X = 500,3073  W = 3073,10

  ## Handling correct scores
  # np.arange creates a matrix of range(num_train). uses that to get the y from each row. with [x,y] x- the row <x> from the scores table for the train sample number,
  # and the column <y> is the correct label for that row
  correct_class_scores = scores[np.arange(num_train), y] # 1,500

  # converts to array of 500,1 . This is to use numpy broadcast to compare each the row of each correct class score TO each column of a row in the training class
  correct_class_scores = correct_class_scores.reshape(len(correct_class_scores),-1)

  # compares a row in the correct class score to each column of a row in the scores matrix
  margins = np.maximum(0, scores - correct_class_scores + 1)

  # setting the score of the correct class to 0
  margins[np.arange(num_train), y] = 0


  ## Calculating the loss
  loss = np.sum(margins)
  loss /= num_train # gets average

  ## Adding regularization to the loss.
  loss += reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  X_mask = np.copy(margins)

  # gradient only with respect to the row of W that corresponds to the correct class
  # column maps to class, row maps to sample
  X_mask[margins > 0] = 1
  # for each sample, find the total number of incorrect classes (margin > 0)
  incorrect_counts = np.sum(X_mask, axis=1)
  X_mask[np.arange(num_train), y] = -incorrect_counts

  dW = np.dot(X.T,X_mask)

  dW /= num_train # average out weights
  dW += reg * 2 * W # regularize the weights
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW


if __name__ == "__main__":

  # Load the raw CIFAR-10 data.
  cifar10_dir = os.path.dirname(
    'C:\\Users\\Matthew\\Documents\\jerusml_deeplearning\\assignments\\spring1617_assignment1\\cs231n\datasets\\cifar-10-python\\')
  X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

  # Split the data into train, val, and test sets. In addition we will
  # create a small development set as a subset of the training data;
  # we can use this for development so our code runs faster.
  num_training = 49000
  num_validation = 1000
  num_test = 1000
  num_dev = 500

  # Our validation set will be num_validation points from the original
  # training set.
  mask = range(num_training, num_training + num_validation)
  X_val = X_train[mask]
  y_val = y_train[mask]

  # Our training set will be the first num_train points from the original
  # training set.
  mask = range(num_training)
  X_train = X_train[mask]
  y_train = y_train[mask]

  # We will also make a development set, which is a small subset of
  # the training set.
  mask = np.random.choice(num_training, num_dev, replace=False)
  X_dev = X_train[mask]
  y_dev = y_train[mask]

  # We use the first num_test points of the original test set as our
  # test set.
  mask = range(num_test)
  X_test = X_test[mask]
  y_test = y_test[mask]


  # Preprocessing: reshape the image data into rows
  X_train = np.reshape(X_train, (X_train.shape[0], -1))
  X_val = np.reshape(X_val, (X_val.shape[0], -1))
  X_test = np.reshape(X_test, (X_test.shape[0], -1))
  X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

  # Preprocessing: subtract the mean image
  # first: compute the image mean based on the training data
  mean_image = np.mean(X_train, axis=0)

  # second: subtract the mean image from train and test data
  X_train -= mean_image
  X_val -= mean_image
  X_test -= mean_image
  X_dev -= mean_image

  # third: append the bias dimension of ones (i.e. bias trick) so that our SVM
  # only has to worry about optimizing a single weight matrix W.
  X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
  X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
  X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
  X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

  # In the file linear_classifier.py, implement SGD in the function
  # LinearClassifier.train() and then run it with the code below.
  from cs231n.classifiers import LinearSVM

  svm = LinearSVM()
  loss_hist = svm.train(X_train, y_train, learning_rate=1e-7, reg=2.5e4,
                        num_iters=1500, verbose=True)


  # Write the LinearSVM.predict function and evaluate the performance on both the
  # training and validation set
  y_train_pred = svm.predict(X_train)
  print('training accuracy: %f' % (np.mean(y_train == y_train_pred), ))
  y_val_pred = svm.predict(X_val)
  print('validation accuracy: %f' % (np.mean(y_val == y_val_pred), ))