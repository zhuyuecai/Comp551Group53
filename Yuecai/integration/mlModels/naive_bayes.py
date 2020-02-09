# Importing necessary libraries...
import collections
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from math import *
class GaussianNaiveBayes:
    total_prob = None
    mean, variance, n_class = None, None, None
    y_dict = None
    n_features = None
    this_class_prior = None


    # calculate the prior distribution for each class in the label
    # label has to be in ints starts from 0
    def class_prior(self, y):
        y_dict = collections.Counter(y)
        n_class = len(y_dict)
        this_class_prior = np.ones(n_class)
        for i in range(n_class):
            this_class_prior[i] = y_dict[i]/y.shape[0]
        return this_class_prior, n_class, y_dict


    # calculate the conditional mean and variance
    def mean_variance(self,X, y):
        n_features = X.shape[1]
        m = np.ones((self.n_class, n_features))
        v = np.ones((self.n_class, n_features))
        xs = []
        for c in range(self.n_class):
            xs.append( np.array([X[i] for i in range(X.shape[0]) if y[i] == c]))
        #xs = np.array(xs)
        for c in range(self.n_class):
            for j in range(n_features):
                m[c][j] = np.mean(xs[c].T[j])
                v[c][j] = np.var(xs[c].T[j], ddof=1)
                if v[c][j] == 0: print((c,j))
        return m, v, n_features # mean and variance 



    def prob_feature_class(self, x):
        m = self.mean
        v = self.variance
        n_sample = x.shape[0]
        pfc = np.ones(( n_sample, self.n_class))
        for s in range(n_sample):
            for i in range(self.n_class):
                pfc[s][i] = np.prod([(1/sqrt(2*3.14*v[i][j])) * exp(-0.5* pow((x[s][j] - m[i][j]),2)/v[i][j])  for j in
                                     range(self.n_features)])
        return pfc


    def fit(self,X, y):
        self.this_class_prior, self.n_class, self.y_dict = self.class_prior(y)
        self.mean, self.variance, self.n_features = self.mean_variance(X, y)


    def predict(self, x):
        n_sample = x.shape[0]
        pfc = self.prob_feature_class(x)
        pcf = np.ones(( n_sample, self.n_class))
        for s in range(n_sample):
            total_prob = 0
            for i in range(self.n_class):
                total_prob = total_prob + (pfc[s][i] * self.this_class_prior[i])
            for i in range(self.n_class):
                pcf[s][i] = (pfc[s][i] * self.this_class_prior[i])/total_prob
        prediction = [int(pcf[s].argmax()) for s in range(n_sample)]
        return prediction




if __name__=="__main__":
    iris = datasets.load_iris()
    print(iris.data[:5])
    print(iris.target[:5])
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data[iris.target < 2 ], iris.target[iris.target < 2], test_size=0.33)
    naive_bayes = GaussianNaiveBayes()
    naive_bayes.fit(X_train, y_train)
    print(naive_bayes.mean)
    print(naive_bayes.variance)
    print(naive_bayes.this_class_prior)
    prid, score = naive_bayes.predict(X_test)
    print(prid)
    print(score)
    print(y_test)
    from sklearn.metrics import average_precision_score
    average_precision = average_precision_score(y_test, score[:,1])

    print('Average precision-recall score: {0:0.2f}'.format(
              average_precision))

