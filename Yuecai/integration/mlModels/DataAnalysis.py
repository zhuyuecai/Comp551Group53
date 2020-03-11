import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class DataProcess
    def __init__(self, data_name):
        self.data_name = data_name


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
        plt.savefig("%s_dis.png"%(self.data_name))
        plt.savefig(data_name)
        plt.close()

    def getHeatMap(self, adata):
        correlation = adata.corr()
        plt.figure(figsize=(12, 12))
        ax = sns.heatmap(
            correlation, annot=True, linewidths=0.5, linecolor="white", vmin=-0.7, cmap="PuOr"
        )
        ax.set_ylim(8.0, 0)
        plt.savefig("%s_heatmap.png"%(self.data_name))
        plt.close()



