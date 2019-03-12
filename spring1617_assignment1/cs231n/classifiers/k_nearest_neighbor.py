import numpy as np
from past.builtins import xrange
import math
from scipy import stats

class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in xrange(num_test):
            for j in xrange(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension.                                    #
                #####################################################################

                squareOfDiff = np.square(X[i] - self.X_train[j])
                dists[i, j] = np.sqrt(np.sum(squareOfDiff))
                '''
                even though this was what the question asked for, it is not going to be used 
                due to slow running time 
                
                rowSum = 0
                for element in range(X[i].size):
                    rowSum += (X[i][element] - self.X_train[j][element])**2
                dists[i][j] = math.sqrt(rowSum)
                '''
                #####################################################################
                #                       END OF YOUR CODE                            #
                #####################################################################

        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in xrange(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            #######################################################################
            testRow = X[i]  # shape (3072)
            squareOfDiff = np.square(self.X_train - testRow)  # shape (5000,3072)
            dists[i, :] = np.sqrt(np.sum(squareOfDiff, axis=1))  # dists.shape: (500,5000), dists[i,:].shape (5000,)
            #######################################################################
            #                         END OF YOUR CODE                            #
            #######################################################################
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy.                #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################

        # going for: (a-b)^2 = a^2 -2ab +b^2:
        a = X
        b = self.X_train

        aSq = a*a
        aSqSum = np.sum(aSq, axis=1).reshape((aSq.shape[0], 1)) #axis 0 is on columns and axis 1 is on rows

        bSq = b*b
        bSqSum = np.sum(bSq, axis=1) #axis 0 is on columns and axis 1 is on rows

        aDotb = np.dot(a,b.T) # dot product returns a single number

        dists = np.sqrt(aSqSum - 2*aDotb + bSqSum)
        pass
        #########################################################################
        #                         END OF YOUR CODE                              #
        #########################################################################
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in xrange(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            sortedIndicies = np.argsort(dists[i])
            closest_y = self.y_train[sortedIndicies]
            pass
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################

            topKPred = closest_y[:k]
            y_pred[i] = stats.mode(topKPred).mode
            pass
        #########################################################################
        #                           END OF YOUR CODE                            #
        #########################################################################

        return y_pred



def test_main():
    #test
    # Load the raw CIFAR-10 data.
    import os
    import random
    import numpy as np
    from cs231n.data_utils import load_CIFAR10
    import matplotlib.pyplot as plt

    cifar10_dir = os.path.dirname('C:\\Users\\Matthew\\Documents\\jerusml_deeplearning\\assignments\\spring1617_assignment1\\cs231n\datasets\\cifar-10-python\\')
    X_train, y_train, X_test, y_test =  load_CIFAR10(cifar10_dir)
    num_training = 5000
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]

    num_test = 500
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))


    # Create a kNN classifier instance.
    # Remember that training a kNN classifier is a noop:
    # the Classifier simply remembers the data and does no further processing
    classifier = KNearestNeighbor()
    classifier.train(X_train, y_train)

############## TEST CODE FROM HERE  ##########---------------------------------------------################################

    num_folds = 5
    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

    X_train_folds = []
    y_train_folds = []
    ################################################################################
    # TODO:                                                                        #
    # Split up the training data into folds. After splitting, X_train_folds and    #
    # y_train_folds should each be lists of length num_folds, where                #
    # y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
    # Hint: Look up the numpy array_split function.                                #
    ################################################################################
    X_train_folds = np.split(X_train, num_folds)
    y_train_folds = np.split(y_train, num_folds)

    pass
    ################################################################################
    #                                 END OF YOUR CODE                             #
    ################################################################################

    # A dictionary holding the accuracies for different values of k that we find
    # when running cross-validation. After running cross-validation,
    # k_to_accuracies[k] should be a list of length num_folds giving the different
    # accuracy values that we found when using that value of k.
    k_to_accuracies = {}

    ################################################################################
    # TODO:                                                                        #
    # Perform k-fold cross validation to find the best value of k. For each        #
    # possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
    # where in each case you use all but one of the folds as training data and the #
    # last fold as a validation set. Store the accuracies for all fold and all     #
    # values of k in the k_to_accuracies dictionary.                               #
    ################################################################################

    numOfFoldsList = range(num_folds)

    for k in k_choices:
        accuracy = []
        for i in numOfFoldsList:
            filteredList = list(filter(lambda x: x != i, numOfFoldsList))

            cvTrainX = [X_train_folds[j] for j in filteredList]
            cvTrainX = np.concatenate(cvTrainX, axis=0)
            cvTrainY = [y_train_folds[j] for j in filteredList]
            cvTrainY = np.concatenate(cvTrainY, axis=0)

            cvTestX = X_train_folds[i]
            cvTestY = y_train_folds[i]

            classifier = KNearestNeighbor()

            classifier.train(cvTrainX, cvTrainY)
            y_test_pred = classifier.predict(cvTestX, k=k)

            num_correct = np.sum(y_test_pred == cvTestY)
            accuracy.append(float(num_correct) / y_test_pred.size)

        k_to_accuracies[k] = accuracy

    ################################################################################
    #                                 END OF YOUR CODE                             #
    ################################################################################

    # Print out the computed accuracies
    for k in sorted(k_to_accuracies):
        for accuracy in k_to_accuracies[k]:
            print('k = %d, accuracy = %f' % (k, accuracy))



#test_main()