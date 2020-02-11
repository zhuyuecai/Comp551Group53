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
#------------------------------
#IMPORT adult data & adult test
#------------------------------
target_col = "income"

adata = pd.read_table(
    "adult.data",
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
)
atest = pd.read_csv(
    "adulttest.txt",
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
adatauncategorical=adata
#------------------------------            
# Get basic statistics:
#------------------------------
print("Analyze the data first \n")
print("Shape is: " ,adata.shape)
print("First few rows:\n", adata.head(5), "\n Averages for the columns:")
print(adata.describe())
#------------------------------
# deal with nulls & other formatting issues
#------------------------------
adata=adata.replace(" ?",np.NaN)
anyNull(adata)
print("Get rid off all null rows\n")
adata= adata.dropna(how='any',axis=0) 
print("New shape is: " ,adata.shape)


# convert variables from int to float and make them categorical
for col in set(adata.columns):
    try:
        adata[col] = adata[col].astype("float")
    except: 
        adata[col] = adata[col].astype("category").cat.codes
        
#shuffle data
adata = adata.sample(frac=1).reset_index(drop=True)

#------------------------------
#FEATURE ANALYSIS
#------------------------------
correlation = adata.corr()
plt.figure(figsize=(12, 12))
print("Also let's check the correlation, to see if there are any patterns to note")
ax=sns.heatmap(correlation,annot=True, linewidths=0.5, linecolor="white",vmin=-0.7, cmap="PuOr")

ax.set_ylim(15, 0) #for heatmap 
plt.show()
print("There aren't many any big correlations")
print("Instead let's check the numerical columns \n")

printGraphs(adata) #columns one by one
plt.show() 

#check gender race income bias
print("Are there any gender / race related biases? \n")
plt.figure(figsize=(14, 12))
ax = sns.catplot(x="race",  hue="gender", data=adatauncategorical, col="income",
                 kind="count")              
plt.show()


#------------------------------
#FEATURE SUBSET
#------------------------------
print("Let's check if we can get rid off any features (feature subset) \n")

plt.figure(figsize=(12, 6))
ax= sns.barplot(data=adatauncategorical, x="education", y="educational-num", )
plt.show()
print("Educational-num and education mean more or less the same thing, so drop education")
adataimproved= adata.drop(columns="education")

#native country
plt.figure(figsize=(12, 6))
sns.countplot(y='native-country', hue='income', data=adatauncategorical)
plt.show()
print("Drop native country as well, as most there are", adataimproved["native-country"].nunique(), "countries, but most entries are from US")
adataimproved= adata.drop(columns="native-country")

#fnlgtw
print("The final weight determined by the Census Organization is of no use in any of the analysis that we are doing henceforth and is removed \n")
adataimproved= adata.drop(columns="fnlwgt")


#for capital loss & gain we have way too many zero's
#capital gain
#adataimproved["capital-gain"]= np.log(adataimproved["capital-gain"]+1) 
#sns.distplot(adataimproved["capital-gain"])
#plt.show()


#------------------------------
#NUMPY CONVERSIONS
#------------------------------
print("Finally, convert to a Numpy arrays")
train=adata.sample(frac = 0.8)
traindata=np.array(train.iloc[:, :-1])
traintarget=np.array(train["income"])

test= adata.drop(train.index)
testdata=np.array(test.iloc[:, :-1])
testtarget=np.array(test["income"])


# Show number of training and testing data points
print("Train segment has size:", traindata.shape)
print("Test segment has size:",testdata.shape)

trainimpr=adataimproved.sample(frac = 0.8)
traindataimpr=np.array(train.iloc[:, :-1])
traintargetimpr=np.array(train["income"])

testimpr= adataimproved.drop(train.index)
testdataimpr=np.array(test.iloc[:, :-1])
testtargetimpr=np.array(test["income"])
