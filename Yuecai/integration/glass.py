import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def anyNull(data):
    print("Number of null values in each column: ")
    for col in data.columns:
        print(col, ":", data[col].isnull().sum())
    return

#import data
idata = pd.read_table("glass.data", sep=',', header=None, names=['id', 'RI', 'Na', 'Mg', 'Al', 'Si','K','Ca','Ba','Fe', 'glasstype'])

#convert variables from int to float and make them categorical
for col in set(idata.columns) - set(idata.describe().columns):
    idata[col] = idata[col].astype('category')
          
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
plt.show()
print("There is a positive correlation between Refractive Index and Calcium, negative one with Silicon")
print("Magnesium has a negative correlation with Calcium, Aluminum, and Barium")
print("Aluminum has a slight positive correlation with Barium\n")

g = idata[['RI', 'Ca']]
gridA = sns.JointGrid(x="Ca", y="RI", data=g, size=6)
gridA=gridA.plot(sns.regplot, sns.distplot)
plt.show()
g = idata[['Mg', 'Ba']]
gridA = sns.JointGrid(x="Ba", y="Mg", data=g, size=6)
gridA=gridA.plot(sns.regplot, sns.distplot)
plt.show()

print("Finally, convert to a Numpy arrays")
train=idata.sample(frac = 0.8)
traindata=np.array(train.iloc[:, :-1])
traintarget=np.array(train["glasstype"])

test= idata.drop(train.index)
testdata=np.array(test.iloc[:, :-1])
testtarget=np.array(test["glasstype"])
# Show number of training and testing data points
print("Train segment has size:", traindata.shape)
print("Test segment has size:", testdata.shape)

#possible feature subsets: add ri calcium, get rid off K, Fe, Na?
