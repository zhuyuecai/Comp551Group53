import numpy as np
from math import *

class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.verbose = verbose
    
    def add_inter(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    

    def fit(self, X, y, eps=0.01):
        X = self.add_inter(X)
        
        # weights initialization
        self.weight = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.weight)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.weight -= self.lr * gradient
            z = np.dot(X, self.weight)
            h = self.__sigmoid(z)
            if self.__loss(h,y) < eps: 
                print("num of iteration %s"%(i))
                break
    
    def predict_prob(self, X):
        X = self.add_inter(X)
        return self.__sigmoid(np.dot(X, self.weight))
    
    def predict(self, X, threshold=0.5):
        return self.predict_prob(X) >= threshold
