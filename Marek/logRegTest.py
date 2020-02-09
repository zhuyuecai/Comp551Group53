import LogisticRegression as lr
import ModelEvaluation as me
import numpy as np


logR = lr.LogisticRegression()

X = np.array([[4], 
              [3], 
              [5], 
              [7]])
y = np.array([[1], 
              [1], 
              [1], 
              [0]])

X_test = np.array([[1],
                   [3],
                   [4],
                   [5],
                   [6],
                   [8],
                   [9]])

y_test = np.array([[1],
                   [1],
                   [1],
                   [0],
                   [1],
                   [0],
                   [0]])

w = logR.fit(X,y)
p = logR.predict(X_test,w)

print('w')
print(w)
print()
print('Predictions')
print(p)
print()
e = me.evaluate_acc(y_test, p)
print('Evaluate')
print(e)

'''
print(logR.predict(X_test))
print(logR.evaluate_acc([1,0,0,1],[0,0,1,1]))


# Data taken from the slides of lecture 3
#
# w0 = 1.05881341
# w1 = 1.61016842
X = np.array([[0.86],[0.09],[-0.85],[0.87],[-0.44],[-0.43],[-1.1],[0.4],[-0.96],[0.17]])
y = np.array([[2.49],[0.83],[-0.25],[3.10],[0.87],[0.02],[-0.12],[1.81],[-0.83],[0.43]])

logR.fit(X, y)
'''
print('End test')


