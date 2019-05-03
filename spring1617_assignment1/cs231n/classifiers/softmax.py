import numpy as np
from random import shuffle
from past.builtins import xrange

from cs231n.data_utils import load_CIFAR10
import os

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    num_train = X.shape[0]
    numClass = W.shape[1]

    # Compute vector of scores
    for i in xrange(num_train):

        scores = X[i].dot(W)  # this is the prediction of training sample i, for each class

        # Normalization trick to avoid numerical instability, per http://cs231n.github.io/linear-classify/#softmax
        scores -= np.max(scores)  # This simply states that we should shift the values inside the vector f so that the highest value is zero

        scores_expsum = np.sum(np.exp(scores))
        cor_ex = np.exp(scores[y[i]])

        # calculate the probabilities that the sample belongs to each class
        allClassProbabilities = np.exp(scores) / scores_expsum
        loss += -np.log(cor_ex / scores_expsum)

        # grad
        # for correct class
        dW[:, y[i]] += (-1) * (scores_expsum - cor_ex) / scores_expsum * X[i]
        for j in xrange(numClass):
            # pass correct class gradient
            if j == y[i]:
                continue
            # for incorrect classes
            dW[:, j] += np.exp(scores[j]) / scores_expsum * X[i]

        '''
        # calculate the gradient for the softmax
        for j in range(numClass):
            dW[:, j] += (probabilities - (j == y[i])) * X[i,:]
        '''

    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    # adding regularization to the gradient
    dW += reg * 2 * W

    pass
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    num_train = X.shape[0]
    f = X.dot(W)

    # Normalization trick to avoid numerical instability, per http://cs231n.github.io/linear-classify/#softmax
    f -= np.max(f, axis=1, keepdims=True)   # This simply states that we should shift the values inside the vector f so that the highest value is zero

    sum_f = np.sum(np.exp(f), axis=1, keepdims=True)
    p = np.exp(f) / sum_f

    loss = np.sum(-np.log(p[np.arange(num_train), y]))

    ind = np.zeros_like(p)
    ind[np.arange(num_train), y] = 1
    dW = X.T.dot(p - ind)

    loss /= num_train
    loss += 2 * reg * np.sum(W * W)
    dW /= num_train
    dW += reg * W

    pass
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW



if __name__ == '__main__':


    def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000, num_dev=500):
        """
        Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
        it for the linear classifier. These are the same steps as we used for the
        SVM, but condensed to a single function.
        """
        # Load the raw CIFAR-10 data
        cifar10_dir = os.path.dirname(
            'C:\\Users\\Matthew\\Documents\\jerusml_deeplearning\\assignments\\spring1617_assignment1\\cs231n\datasets\\cifar-10-python\\')
        X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

        # subsample the data
        mask = list(range(num_training, num_training + num_validation))
        X_val = X_train[mask]
        y_val = y_train[mask]
        mask = list(range(num_training))
        X_train = X_train[mask]
        y_train = y_train[mask]
        mask = list(range(num_test))
        X_test = X_test[mask]
        y_test = y_test[mask]
        mask = np.random.choice(num_training, num_dev, replace=False)
        X_dev = X_train[mask]
        y_dev = y_train[mask]

        # Preprocessing: reshape the image data into rows
        X_train = np.reshape(X_train, (X_train.shape[0], -1))
        X_val = np.reshape(X_val, (X_val.shape[0], -1))
        X_test = np.reshape(X_test, (X_test.shape[0], -1))
        X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

        # Normalize the data: subtract the mean image
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image
        X_dev -= mean_image

        # add bias dimension and transform into columns
        X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
        X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
        X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
        X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

        return X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev


    # Invoke the above function to get our data.
    X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = get_CIFAR10_data()
    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', X_val.shape)
    print('Validation labels shape: ', y_val.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)
    print('dev data shape: ', X_dev.shape)
    print('dev labels shape: ', y_dev.shape)

    #from cs231n.classifiers.softmax import softmax_loss_naive
    import time

    # Generate a random softmax weight matrix and use it to compute the loss.
    W = np.random.randn(3073, 10) * 0.0001
    #loss, grad = softmax_loss_naive(W, X_dev, y_dev, 0.0)
    loss, grad =  loss_vectorized, grad_vectorized = softmax_loss_vectorized(W, X_dev, y_dev, 0.0)

    # As a rough sanity check, our loss should be something close to -log(0.1).
    print('loss: %f' % loss)
    print('sanity check: %f' % (-np.log(0.1)))
