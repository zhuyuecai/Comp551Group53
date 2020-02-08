import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def anyNull(data):
    counter=0
    for col in data.columns:
        counter= counter+data[col].isnull().sum()
    print("There are", counter, "null values.")
    return counter

def NullDetailed(data):
    print("Number of null values in each column: ")
    for col in data.columns:
        print(col, ":", data[col].isnull().sum())
    return

  
#import ionosphere data
idata = pd.read_table("ionosphere.data", sep=',', header=None)

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

correlation = idata.corr()
plt.figure(figsize=(12, 12))
print("Also let's check the correlation, to see if there are any patterns to note")
print("Correlation heatmapt might take a while")
heatmap = sns.heatmap(correlation, linewidths=0.5, linecolor="white",vmin=-0.7, cmap="PuOr")

#Check for nulls
nullamount=anyNull(idata)
if nullamount >0 : NullDetailed(idata) 
            
#row one contains only 0's, so we can remove the whole row
print("\n Rows 1 contains only 0s, we can remove it \n")

idata=idata.drop(columns=[1])

print("Finally, convert to a Numpy array")
idata=idata.to_numpy()

print("Split into data and target segments")
#Last column is the target (boolean t or f)
data = idata[:, :-1]
target = idata[:, -1:]

# Show number of training and testing data points
print("Data segment has size:", data.shape)
print("Target segment has size:",target.shape)
