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
    plt.subplot(2, 4, 1)
    sns.distplot(adata['mpg'])
    plt.subplot(2, 4,2)
    sns.distplot(adata['cylinders'])
    plt.subplot(2, 4, 3)
    sns.distplot(adata['horsepower'])
    plt.subplot(2, 4, 4)
    sns.distplot(adata['weight'])
    plt.subplot(2, 4, 5)
    sns.distplot(adata['acceleration'])
    plt.subplot(2, 4, 6)
    sns.distplot(adata['model year'])
    plt.subplot(2, 4, 7)
    sns.distplot(adata['origin'])
  
#import ionosphere data
adata = pd.read_table("auto-mpg.data", delim_whitespace=True, header=None, names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight','acceleration', 'model year', 'origin', 'car name'])

#convert variables from int to float and make them categorical
for col in set(adata.columns) - set(adata.describe().columns):
    adata[col] = adata[col].astype('category')
          
# Get basic statistics:
print("Analyze the data first \n")
print("Shape is: " ,adata.shape)
print("First few rows:\n", adata.head(5), "\n Averages for the columns:")
print(adata.describe())


adata.horsepower=adata.horsepower.replace('?',float('nan'))
adata.horsepower = adata.horsepower.astype(float) #horsepower is an object, need to convert to a number
anyNull(adata)
print("Get rid off all null rows in horsepower\n")
adata= adata.dropna(how='any',axis=0) 
print("New shape is: " ,adata.shape)

correlation = adata.corr()
plt.figure(figsize=(12, 12))
print("Also let's check the correlation, to see if there are any patterns to note")
ax=sns.heatmap(correlation,annot=True, linewidths=0.5, linecolor="white",vmin=-0.7, cmap="PuOr")
ax.set_ylim(8.0, 0)
plt.show()
print("There are high correlations between cylinders, displacement, horsepower, and weight\n")
print("We also have high negative correlations between mpg and cylinders, displacement, horsepower, weight\n")
printGraphs(adata) #fork in an ideal world but os module doesn't work on Windows

print("Finally, convert to a Numpy arrays")
train=adata.sample(frac = 0.8)
traindata=np.array(train.iloc[:, 1:])
traintarget=np.array(train["mpg"])

test= adata.drop(train.index)
testdata=np.array(test.iloc[:, 1:])
testtarget=np.array(test["mpg"])
# Show number of training and testing data points
print("Train segment has size:", traindata.shape)
print("Test segment has size:", testdata.shape)

#possible feature sets for later: force (weight * acceleration), weight horsepower?
