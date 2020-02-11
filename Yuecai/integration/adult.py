import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlModels.naive_bayes import GaussianNaiveBayes
from mlModels.NewLogistic import LogisticRegression
from mlModels.CrossValidation import Cross_Validation
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

# naive_experiment for accuracy 
naive_bayes = GaussianNaiveBayes()
train_score, test_score = Cross_Validation.Cross_Validation(naive_bayes, 5, np.array(adata.iloc[:, :-1]),
                                                            np.array(adata[target_col]))
print('naive_bayes accuracy score: {0:0.2f}'.format(np.mean(test_score)))

# naive_experiment for accuracy 
logres = LogisticRegression(num_iter=10000)
train_score, test_score = Cross_Validation.Cross_Validation(logres, 5, np.array(adata.iloc[:, :-1]),
                                                           np.array(adata[target_col]))
print('logistic regression accuracy score: {0:0.2f}'.format(np.mean(test_score)))


naive_test_score = []
logi_test_score = []
sample_size = list(range(2,100,2))
#change number of folds to control the training sample size
for k in sample_size:
    train_score, test_score = Cross_Validation.Size_Experiment(naive_bayes, k, np.array(adata.iloc[:, :-1]),
                                                               np.array(adata.iloc[: ,-1:]))
    naive_test_score.append(np.mean(test_score))
    train_score, test_score = Cross_Validation.Size_Experiment(logres, k, np.array(adata.iloc[:, :-1]),
                                                               np.array(adata.iloc[: ,-1:]))
    logi_test_score.append(np.mean(test_score))

dd =pd.DataFrame(list(zip(sample_size, naive_test_score,logi_test_score)),
                 columns=["sample_size","NB","Logit"])
plt.figure(figsize=(10, 5))
plt.subplot(1,2,1)
sns.lineplot(x="sample_size",y="NB", data=dd)
plt.subplot(1,2,2)
sns.lineplot(x="sample_size",y="Logit", data=dd)
plt.savefig("adult_samplesize.png")





