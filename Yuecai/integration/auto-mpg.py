import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlModels.naive_bayes import GaussianNaiveBayes
#from mlModels.LogisticRegression import LogisticRegression
from mlModels.NewLogistic import LogisticRegression
from mlModels.CrossValidation import Cross_Validation
from sklearn.metrics import average_precision_score, accuracy_score

def anyNull(data):
    print("Number of null values in each column: ")
    for col in data.columns:
        print(col, ":", data[col].isnull().sum())
    return

# import ionosphere data
adata = pd.read_table(
    "data_sets/auto-mpg.data",
    delim_whitespace=True,
    header=None,
    names=[
        "mpg",
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model year",
        "origin",
        "car name",
    ],
    converters={'mpg': lambda x: int(float(x) > 23.5)}
)

# convert variables from int to float and make them categorical
for col in set(adata.columns):
    try:
        adata[col] = adata[col].astype("float")
    except: 
        adata[col] = adata[col].astype("category").cat.codes


print("Finally, convert to a Numpy arrays")
train = adata.sample(frac=0.8)
traindata = np.array(train.iloc[:, 1:])
traintarget = np.array(train["mpg"])

test = adata.drop(train.index)
testdata = np.array(test.iloc[:, 1:])
testtarget = np.array(test["mpg"])
# Show number of training and testing data points
print("Train segment has size:", traindata.shape)
print("Test segment has size:", testdata.shape)
print("Train segment target has size:", traintarget.shape)
print("Test segment target has size:", testtarget.shape)
# possible feature sets for later: force (weight * acceleration), weight horsepower?

# naive_experiment for accuracy 
naive_bayes = GaussianNaiveBayes()
train_score, test_score = Cross_Validation.Cross_Validation(naive_bayes, 5, np.array(adata.iloc[:, 1:]),
                                                           np.array(adata["mpg"]))
print('naive_bayes accuracy score: {0:0.2f}'.format(
              np.mean(test_score)))

# naive_experiment for accuracy 
logres = LogisticRegression()
train_score, test_score = Cross_Validation.Cross_Validation(logres, 5, np.array(adata.iloc[:, 1:]),
                                                           np.array(adata["mpg"]))
print('logistic regression accuracy score: {0:0.2f}'.format(
              np.mean(test_score)))
