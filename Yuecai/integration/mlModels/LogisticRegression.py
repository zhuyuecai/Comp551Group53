import numpy as np
import math

# ---------------------------------
# Compute logistic regression
#
# author: Marek Adamowicz
# since : Feb 7 2020
# ---------------------------------
class LogisticRegression:
    # w = None

    # --------------------------------------------------
    # Fit using method from Lecture 3 (Linear Regression)
    #
    # input:  X, y = training data
    # output: None
    # effect: Modification of model parameters
    # --------------------------------------------------
    def linRegFit(self, X, y):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        w = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))
        return w

    # --------------------------------------------------
    # Compute formula for a single value
    #
    # input:  a = single value
    # output: result of applying the function
    # --------------------------------------------------
    def logistic(self, a):
        return 1 / (1 + math.exp(-a))

    # --------------------------------------------------
    # Compute the logistic/sigmoid function
    #
    # input:  Value to apply sigmoid
    # output: Result of sigmoid
    # --------------------------------------------------
    def logisticArray(self, value):
        L = np.zeros((len(value), 1))
        for i in range(0, len(value)):
            L[i] = self.logistic(value[i])
        return L

    # --------------------------------------------------
    # Do an iteration of gradient descent
    #
    # input:  X, y = training data
    # input:  w = model parameters
    # output: descent
    # --------------------------------------------------
    def gradient(self, X, y, w):
        N, D = X.shape
        yh = self.logisticArray(np.dot(X, w)[:,0])
        grad = np.dot(X.T, yh - y) / N
        return grad

    # --------------------------------------------------
    # Fit using method from Lecture 7 (Gradient Descent)
    #
    # Note: This should be full batch SGD, an extra line
    # is needed for minibatch
    #
    # input:  X, y = training data
    # output: None
    # effect: Modification of model parameters
    # --------------------------------------------------
    def fit(self, X, y, lr=0.01, eps=1e-2, max_iter = 1000):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        N, D = X.shape
        w = np.random.rand(D,1)
        #w = np.zeros((D, 1))
        g = np.inf
        c = 1
        while np.linalg.norm(g) > eps and c < max_iter:
            g = self.gradient(X, y, w)
            w = w - lr * g
            c = c + 1
        return w

    # --------------------------------------------------
    # Predicts the value of input points
    #
    # input:  X = input points
    # output: Predictions for these points
    # --------------------------------------------------
    def predict(self, X, w):
        y_predict = [None] * len(X)
        f_x = 0
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        # Go through data and check it is above the line
        # yes = predict 1
        # no  = predict 0
        for i in range(0, len(X)):
            f_x = 0
            for j in range(0, len(X[0])):
                f_x = f_x + w[j, 0] * X[i][j]
            r = self.logistic(f_x)
            print(r)
            if r > 0.5:
                print("t")
                y_predict[i] = 1
            else:
                print("f")
                y_predict[i] = 0

        return y_predict


