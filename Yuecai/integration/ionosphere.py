import math

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlModels.naive_bayes import GaussianNaiveBayes
from mlModels.NewLogistic import LogisticRegression
from mlModels.CrossValidation import Cross_Validation
from sklearn.metrics import average_precision_score, accuracy_score


def anyNull(data):
    counter=0
    for col in data.columns:
        counter= counter+data[col].isnull().sum()
    print("There are", counter, "null values.")
    return counter

  
#import ionosphere data
idata = pd.read_table("data_sets/ionosphere.data", sep=',', header=None)

#convert variables from int to float and make them categorical
for col in idata.columns:
        col_values = idata[col]
        if col_values.dtype == int:
            idata.iloc[:, col] = col_values.astype(float)
        elif col_values.dtype != float:
            idata.iloc[:, col] = col_values.astype('category')
            
# Get basic statistics:
print("Analyze the data first \n")
print("Shape is: " ,idata.shape)
print("First few rows:\n", idata.head(5), "\n Averages for the columns:")
print(idata.describe(include='all'))

#Check for nulls
anyNull(idata)
         
#into Numpy

print("Split into data and target segments")

#column one contains only 0's, so we can try and see what happens if we remove the whole column

for col in set(idata.columns):
    try:
        idata[col] = idata[col].astype("float")
    except: 
        idata[col] = idata[col].astype("category").cat.codes

print("Finally, convert to a Numpy array")
adata=idata

#find correlations
correlation = idata.corr()
plt.figure(figsize=(12, 12))
print("Also let's check the correlation, to see if there are any patterns to note")
#output correlation heatmap
ax=heatmap = sns.heatmap(correlation, linewidths=0.5, linecolor="white",vmin=-0.7, cmap="PuOr")
ax.set_ylim(34.0, 0)
ax.set_xlim(34.0, 0)
plt.savefig("iono_heat.png")
plt.close()
# naive_experiment for accuracy 
naive_bayes = GaussianNaiveBayes()
train_score, test_score = Cross_Validation.Cross_Validation(naive_bayes, 5, np.array(adata.iloc[:, 2:-1]),
                                                           np.array(adata[34]))
print('naive_bayes accuracy score: {0:0.2f}'.format(np.mean(test_score)))
logres = LogisticRegression()
train_score, test_score = Cross_Validation.Cross_Validation(logres, 5, np.array(adata.iloc[:, 2:-1]),
                                                           np.array(adata[34]))
print('logistic regression accuracy score: {0:0.2f}'.format(np.mean(test_score)))

#change number of folds to control the training sample size
naive_test_score = []
logi_test_score = []
sample_size = list(range(2,100,2))
for k in sample_size:
    train_score, test_score = Cross_Validation.Size_Experiment(naive_bayes, k, np.array(adata.iloc[:, 2:-1]),
                                                               np.array(adata.iloc[: ,34:]))
    print('naive_bayes accuracy score: {0:0.2f}'.format(
                  np.mean(test_score)))
    naive_test_score.append(np.mean(test_score))
    train_score, test_score = Cross_Validation.Size_Experiment(logres, k, np.array(adata.iloc[:, 2:-1]),
                                                               np.array(adata.iloc[: ,34:]))
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
plt.savefig("iono_samplesize.png")

train_X = np.array(adata.iloc[:, 2:-1])
train_y = np.array(adata.iloc[: ,34])
plt.figure(figsize=(10, 20))
for lr in range(0, 5):
    lrt = 0.01 + lr*0.04,
    logres_lr = LogisticRegression(lr=lrt, num_iter=2000)
    result = logres_lr.fit_lr_test(train_X, train_y, eps=0.000001)
    plt.subplot(3,2,lr+1)
    plt.plot(result[0], result[1], label=str(lrt))
    plt.title("lrate %s"%(lrt))
plt.savefig("iono_lr.png")
plt.close()









