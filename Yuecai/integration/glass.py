import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlModels.naive_bayes import GaussianNaiveBayes
from mlModels.NewLogistic import LogisticRegression
from mlModels.CrossValidation import Cross_Validation

def anyNull(data):
    print("Number of null values in each column: ")
    for col in data.columns:
        print(col, ":", data[col].isnull().sum())
    return

#import data
idata = pd.read_table("data_sets/glass.data", sep=',', header=None, names=['id', 'RI', 'Na', 'Mg', 'Al', 'Si','K','Ca','Ba','Fe', 'glasstype'])

#convert variables from int to float and make them categorical
#for col in set(idata.columns) - set(idata.describe().columns):
    #idata[col] = idata[col].astype('category')
idata = idata[idata.glasstype <3]
idata["glasstype"] -= 1 

print(idata["glasstype"].unique())         
# Get basic statistics:
print("Analyze the data first \n")
print("Shape is: " ,idata.shape)
print("First few rows:\n", idata.head(5), "\n Averages for the columns:")
print(idata.describe())

#shuffle data
idata = idata.sample(frac=1).reset_index(drop=True)

anyNull(idata)
print("There are no nulls \n")

cidata=idata.drop(columns='id')
correlation = cidata.corr()
plt.figure(figsize=(12, 12))
print("Also let's check the correlation, to see if there are any patterns to note")

ax=sns.heatmap(correlation,annot=True, linewidths=0.5, linecolor="white",vmin=-0.7, cmap="PuOr")
ax.set_ylim(10.0, 0)
plt.savefig("glass_heat.png")
plt.close()
print("There is a positive correlation between Refractive Index and Calcium, negative one with Silicon")
print("Magnesium has a negative correlation with Calcium, Aluminum, and Barium")
print("Aluminum has a slight positive correlation with Barium\n")

g = idata[['RI', 'Ca']]
gridA = sns.JointGrid(x="Ca", y="RI", data=g, size=6)
gridA=gridA.plot(sns.regplot, sns.distplot)
plt.savefig("glass_RI_Ca.png")
plt.close()
g = idata[['Mg', 'Ca']]
gridA = sns.JointGrid(x="Ca", y="Mg", data=g, size=6)
gridA=gridA.plot(sns.regplot, sns.distplot)
plt.savefig("glass_Mg_Ca.png")
plt.close()

# convert variables from int to float and make them categorical
for col in set(idata.columns):
    try:
        idata[col] = idata[col].astype("float")
    except: 
        idata[col] = idata[col].astype("category").cat.codes

print(idata.columns)
#drop ca column
adata = idata.drop(columns = 'RI')
#adata = idata
#possible feature subsets: add ri calcium, get rid off K, Fe, Na?
# naive_experiment for accuracy 
naive_bayes = GaussianNaiveBayes()
train_score, test_score = Cross_Validation.Cross_Validation(naive_bayes, 5, np.array(adata.iloc[:, 1:-1]),
                                                           np.array(adata["glasstype"]))
print('naive_bayes accuracy score: {0:0.2f}'.format(np.mean(test_score)))

# naive_experiment for accuracy 
logres = LogisticRegression()
train_score, test_score = Cross_Validation.Cross_Validation(logres, 5, np.array(adata.iloc[:, 1:-1]),
                                                           np.array(adata["glasstype"]))
print('logistic regression accuracy score: {0:0.2f}'.format(np.mean(test_score)))

#change number of folds to control the training sample size
naive_test_score = []
logi_test_score = []
sample_size = list(range(2,100,2))
for k in sample_size:
    train_score, test_score = Cross_Validation.Size_Experiment(naive_bayes, k, np.array(adata.iloc[:, 1:-1]),
                                                               np.array(adata.iloc[: ,-1:]))
    print('naive_bayes accuracy score: {0:0.2f}'.format(
                  np.mean(test_score)))
    naive_test_score.append(np.mean(test_score))
    train_score, test_score = Cross_Validation.Size_Experiment(logres, k, np.array(adata.iloc[:, 1:-1]),
                                                               np.array(adata.iloc[: ,-1:]))
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
plt.savefig("glass_samplesize.png")

train_X = np.array(adata.iloc[:, 1:-1])
train_y = np.array(adata.iloc[: ,-1])
plt.figure(figsize=(10, 20))
for lr in range(0, 5):
    lrt = 0.01 + lr*0.04,
    logres_lr = LogisticRegression(lr=lrt, num_iter=4000)
    result = logres_lr.fit_lr_test(train_X, train_y, eps=0.000001)
    plt.subplot(3,2,lr+1)
    plt.plot(result[0], result[1], label=str(lrt))
    plt.title("lrate %s"%(lrt))
plt.savefig("glass_lr.png")


