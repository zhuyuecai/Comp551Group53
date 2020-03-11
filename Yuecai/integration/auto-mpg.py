import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlModels.naive_bayes import GaussianNaiveBayes
#from mlModels.LogisticRegression import LogisticRegression
from mlModels.NewLogistic import LogisticRegression
from mlModels.CrossValidation import Cross_Validation

def anyNull(data):
    print("Number of null values in each column: ")
    for col in data.columns:
        print(col, ":", data[col].isnull().sum())
    return

def printGraphs(adata, data_name):
    plt.figure(figsize=(10, 10))
    plt.axis("equal")
    plt.subplot(2, 4, 1)
    sns.distplot(adata["mpg"])
    plt.subplot(2, 4, 2)
    sns.distplot(adata["cylinders"])
    plt.subplot(2, 4, 3)
    sns.distplot(adata["horsepower"])
    plt.subplot(2, 4, 4)
    sns.distplot(adata["weight"])
    plt.subplot(2, 4, 5)
    sns.distplot(adata["acceleration"])
    plt.subplot(2, 4, 6)
    sns.distplot(adata["model year"])
    plt.subplot(2, 4, 7)
    sns.distplot(adata["origin"])
    plt.savefig("%s_dis.png"%(data_name))
    plt.savefig(data_name)
    plt.close()

def getHeatMap(adata, data_name):
    correlation = adata.corr()
    plt.figure(figsize=(12, 12))
    ax = sns.heatmap(
        correlation, annot=True, linewidths=0.5, linecolor="white", vmin=-0.7, cmap="PuOr"
    )
    ax.set_ylim(8.0, 0)
    plt.savefig("%s_heatmap.png"%(data_name))
    plt.close()

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
print("Analyze the data first \n")
print("Shape is: ", adata.shape)
print(adata.describe())
# convert variables from int to float and make them categorical
for col in set(adata.columns):
    try:
        adata[col] = adata[col].astype("float")
    except: 
        adata[col] = adata[col].astype("category").cat.codes

# get rid off null values
adata.horsepower = adata.horsepower.replace("?", float("nan"))
adata.horsepower = adata.horsepower.astype(
        float
)  # horsepower is an object, need to convert to a number
anyNull(adata)
print("Get rid off all null rows in horsepower\n")
adata = adata.dropna(how="any", axis=0)
print("New shape is: ", adata.shape)

getHeatMap(adata, "auto_mpg")
printGraphs(adata, "auto_mpg")


# possible feature sets for later: force (weight * acceleration), weight horsepower?
# naive_experiment for accuracy 
naive_bayes = GaussianNaiveBayes()
train_score, test_score = Cross_Validation.Cross_Validation(naive_bayes, 5, np.array(adata.iloc[:, 1:]),
                                                           np.array(adata["mpg"]))
print('naive_bayes accuracy score: {0:0.2f}'.format(np.mean(test_score)))

# naive_experiment for accuracy 
logres = LogisticRegression()
train_score, test_score = Cross_Validation.Cross_Validation(logres, 5, np.array(adata.iloc[:, 1:]),
                                                           np.array(adata["mpg"]))
print('logistic regression accuracy score: {0:0.2f}'.format(np.mean(test_score)))


#change number of folds to control the training sample size
naive_test_score = []
logi_test_score = []
sample_size = list(range(2,100,2))
for k in sample_size:
    train_score, test_score = Cross_Validation.Size_Experiment(naive_bayes, k, np.array(adata.iloc[:, 1:]),
                                                               np.array(adata.iloc[: ,:1]))
    print('naive_bayes accuracy score: {0:0.2f}'.format(
                  np.mean(test_score)))
    naive_test_score.append(np.mean(test_score))
    train_score, test_score = Cross_Validation.Size_Experiment(logres, k, np.array(adata.iloc[:, 1:]),
                                                               np.array(adata.iloc[: ,:1]))
    print('logistic regression accuracy score: {0:0.2f}'.format(
                  np.mean(test_score)))
    logi_test_score.append(np.mean(test_score))

dd =pd.DataFrame(list(zip(sample_size, naive_test_score,logi_test_score)),
                 columns=["sample_size","NB","Logit"])
                

plt.figure(figsize=(10, 5))
plt.subplot(1,2,1)
sns.lineplot(x="sample_size",y="NB", data=dd)
plt.subplot(1,2,2)
sns.lineplot(x="sample_size",y="Logit", data=dd)
plt.savefig("auto_mpg_samplesize.png")
plt.close()
train_X = np.array(adata.iloc[:, 1:])
train_y = np.array(adata.iloc[: ,0])

plt.figure(figsize=(10, 20))
for lr in range(0, 5):
    lrt = 0.01 + lr*0.04,
    logres_lr = LogisticRegression(lr=lrt, num_iter=2000)
    result = logres_lr.fit_lr_test(train_X, train_y)
    plt.subplot(3,2,lr+1)
    plt.plot(result[0], result[1], label=str(lrt))
    plt.title("lrate %s"%(lrt))
plt.savefig("auto_mpg_lr.png")
plt.close()







