import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def anyNull(data):
    print("Number of null values in each column: ")
    for col in data.columns:
        print(col, ":", data[col].isnull().sum())
    return

def printGraphs(adata):
    plt.figure(figsize=(10, 10))
    plt.axis('equal')
    plt.subplot(2, 3, 1)
    sns.distplot(adata['age'])
    plt.subplot(2, 3, 2)
    sns.distplot(adata['fnlwgt'])
    plt.subplot(2, 3, 3)
    sns.distplot(adata['educational-num'])
    plt.subplot(2, 3, 4)
    sns.distplot(adata['capital-gain'])
    plt.subplot(2, 3, 5)
    sns.distplot(adata['capital-loss'])
    plt.subplot(2, 3, 6)
    sns.distplot(adata['hours-per-week'])
  
#import adult data & adult test
adata = pd.read_table("adult.data", sep=',', header=None, names=['age', 'workclass', 'fnlwgt', 'education', 'educational-num','marital-status', 'occupation', 'relationship', 'race', 'gender','capital-gain', 'capital-loss', 'hours-per-week', 'native-country','income'])
atest = pd.read_csv('adulttest.txt', sep=",\s", header=None, names = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num','marital-status', 'occupation', 'relationship', 'race', 'gender','capital-gain', 'capital-loss', 'hours-per-week', 'native-country','income'], engine = 'python')


#convert variables from int to float and make them categorical
for col in set(adata.columns) - set(adata.describe().columns):
    adata[col] = adata[col].astype('category')
for col in set(atest.columns) - set(atest.describe().columns):
    atest[col] = atest[col].astype('category')
atest['income'].replace(regex=True,inplace=True,to_replace=r'\.',value=r'')
#for some reason test file ended with a '.' for income
            
# Get basic statistics:
print("Analyze the data first \n")
print("Shape is: " ,adata.shape)
print("First few rows:\n", adata.head(5), "\n Averages for the columns:")
print(adata.describe())
print("Looks like we have people working 99 hours a week, which is illegal but possible if one works 14 hours every day of the week")

adata=adata.replace(" ?",np.NaN)
atest=atest.replace(" ?",np.NaN)
anyNull(adata)
print("Get rid off all null rows\n")
adata= adata.dropna(how='any',axis=0) 
atest= atest.dropna(how='any',axis=0)
print("New shape is: " ,adata.shape)

correlation = adata.corr()
plt.figure(figsize=(12, 12))
print("Also let's check the correlation, to see if there are any patterns to note")
ax=sns.heatmap(correlation,annot=True, linewidths=0.5, linecolor="white",vmin=-0.7, cmap="PuOr")
ax.set_ylim(6.0, 0)
plt.show()
print("There aren't any useful correlations\n Let's check for any outliers in each column \n")
printGraphs(adata) #fork in an ideal world but os module doesn't work on Windows
plt.show()

print("Finally, convert to a Numpy arrays")
traindata=np.array(adata.iloc[:, :-1])
traintarget=np.array(adata["income"])
testdata=np.array(atest.iloc[:, :-1])
testtarget=np.array(atest["income"])
# Show number of training and testing data points
print("Train segment has size:", traindata.shape)
print("Test segment has size:",testdata.shape)

#TODO: more interesting graphs like race vs gender vs hours / week
