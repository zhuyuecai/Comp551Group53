import numpy as np

# ---------------------------------
# Compute logistic regression
#
# author: Marek Adamowicz
# since : Feb 9 2020
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
        X = np.hstack((np.ones((X.shape[0],1)),X))
        w = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))
        return w


    # --------------------------------------------------
    # Compute the sigmoid function for a single value
    #
    # input:  a = single value
    # output: result of applying the function
    # --------------------------------------------------
    def sigmoid(self, a):
        return 1 / (1 + np.exp(-a))
    
    
    # --------------------------------------------------
    # Compute the sigmoid function for an array
    #
    # input:  Value to apply sigmoid
    # output: Result of sigmoid
    # --------------------------------------------------
    def logistic(self, X, w):
        return self.sigmoid(np.dot(X,w))
    
    # --------------------------------------------------
    # Compute the loss function to determine which
    # way to take the gradient descent
    #
    # input: X, y = training data
    # input: w = model parameters
    # --------------------------------------------------
    def cost(self, X, y, w):
        z = np.dot(X,w)
        J = np.mean(y * np.log1p(np.exp(-z)) + (1-y) * np.log1p(np.exp(z)))
        return J
    

    # --------------------------------------------------
    # Do an iteration of gradient descent
    #
    # input:  X, y = training data
    # input:  w = model parameters
    # output: descent
    # --------------------------------------------------
    def gradient(self, X, y, w):
        N,D = X.shape        
        yh = self.logistic(X,w)
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
    def fit(self, X, y, lr=0.1, eps=1e-2, num_iter=300000):
        X = np.hstack((np.ones((X.shape[0],1)),X)) # Add Intercept
        N,D = X.shape
        # w = np.zeros((D, 1))          # <- Super Slow
        #w = np.zeros(X.shape[1])       # <- Super Fast
        w = np.random.rand(X.shape[1])  # <- Fast and random
        w = w * 50
        g = np.inf
        iterations = 0
        while np.linalg.norm(g) > eps:
            g = self.gradient(X, y, w)
            w = w - lr*g
            iterations += 1
            if iterations == num_iter:
                #print('Iterations %d' % (iterations))
                return w
        #print('Iterations: %d' % (iterations))
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
        X = np.hstack((np.ones((X.shape[0],1)),X))
        
        # Predict 1 or 0 for each data point
        for i in range(0, len(X)):
            f_x = 0  
            for j in range(0, len(X[0])):
                f_x = f_x + w[j] * X[i][j]    
            r = self.sigmoid(f_x)
            if r > 0.5:
                y_predict[i] = 1
            else:
                y_predict[i] = 0

        return y_predict