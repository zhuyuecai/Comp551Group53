import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlModels.naive_bayes import GaussianNaiveBayes
from sklearn.metrics import average_precision_score, accuracy_score

target_col = "income"

def anyNull(data):
    print("Number of null values in each column: ")
    for col in data.columns:
        print(col, ":", data[col].isnull().sum())
    return


# import adult data & adult test
adata = pd.read_table(
    "data_sets/adult.data",
    sep=",",
    header=None,
    names=[
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "educational-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "gender",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ],
)
atest = pd.read_csv(
    "data_sets/adult.test",
    sep=",\s",
    header=None,
    names=[
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "educational-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "gender",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ],
    engine="python"
)


# for some reason test file ended with a '.' for income
atest["income"].replace(regex=True, inplace=True, to_replace=r"\.", value=r"")

adata = pd.concat([adata,atest])
#drop it since 0 variance, this feature is a strong signal, can be used rule based to assemble the result
adata["income"].replace(regex=True, inplace=True, to_replace=r" ", value=r"")
adata[target_col].replace(' ', '', inplace=True)
# Get basic statistics:
print("Analyze the data first \n")
print("Shape is: ", adata.shape)
print("First few rows:\n", adata.head(5), "\n Averages for the columns:")
print(adata.describe())
print(
    "Looks like we have people working 99 hours a week, which is illegal but possible if one works 14 hours every day of the week"
)

adata = adata.replace(" ?", np.NaN)
anyNull(adata)
print("Get rid off all null rows\n")
adata = adata.dropna(how="any", axis=0)
print("New shape is: ", adata.shape)

# convert variables from int to float and make them categorical
for col in set(adata.columns):
    try:
        adata[col] = adata[col].astype("float")
    except: 
        adata[col] = adata[col].astype("category").cat.codes

print("Finally, convert to a Numpy arrays")
train = adata.sample(frac=0.8)
traindata = np.array(train.iloc[:, :-1])
traintarget = np.array(train[target_col])

test = adata.drop(train.index)
testdata = np.array(test.iloc[:, :-1])
testtarget = np.array(test[target_col])
# Show number of training and testing data points
print("Train segment has size:", traindata.shape)
print("Test segment has size:", testdata.shape)
print("Train segment target has size:", traintarget.shape)
print("Test segment target has size:", testtarget.shape)


# naive_experiment for accuracy 
naive_bayes = GaussianNaiveBayes()
naive_bayes.fit(traindata, traintarget)
pre, score = naive_bayes.predict(testdata)
average_precision = average_precision_score(testtarget, score[:,1])
accuracy = accuracy_score(testtarget, pre)
print('Average precision-recall score: {0:0.2f}'.format(
              average_precision))
print('accuracy score: {0:0.2f}'.format(
              accuracy))
