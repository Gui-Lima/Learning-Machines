import pandas as pd
import sys
from os import path
import matplotlib.pyplot
import numpy as np

sys.path.insert(0, path.abspath('Separating Data/kFoldCrossValidation/'))
sys.path.insert(1, path.abspath('DataSets/Class-level data for KC1 (Defective or Not) - Software defect prediction/'))
sys.path.insert(2, path.abspath('KNN/'))
sys.path.insert(3, path.abspath('DataSets/'))

from kCrossValidation import kCrossValidation as kc
import ReadingIt as rd
from KNN import Knn
import Reading as r


def simpleKnn():
    dataSet = r.readDataSet(rd.relPath, rd.columns)
    groups = dataSet['DL'].tolist()
    trainingSets = []
    avaliationSets = []
    kc(5, groups, trainingSets, avaliationSets)
    dataSet = dataSet.apply(pd.to_numeric)

    for i in range(len(trainingSets)):
        tset=[]
        aset=[]
        for index, row in dataSet.iterrows():
            tupla = (dataSet.iloc[index]['DL'], index)
            if tupla in trainingSets[i]:
                tset.append(row.tolist())
            if tupla in avaliationSets[i]:
                aset.append(row.tolist())
        k = Knn(tset, 1)
        k.test(aset)

def makeGraph():
    dataSet = r.readDataSet(rd.relPath, rd.columns)
    groups = dataSet['DL'].tolist()
    trainingSets = []
    avaliationSets = []
    kc(5, groups, trainingSets, avaliationSets)
    dataSet = dataSet.apply(pd.to_numeric)

    ks = [1,2,3,5,7,9,11,13,15]
    means = []
    for j in ks:
        correctPercentage = 0
        for i in range(len(trainingSets)):
            tset=[]
            aset=[]
            for index, row in dataSet.iterrows():
                tupla = (dataSet.iloc[index]['DL'], index)
                if tupla in trainingSets[i]:
                    tset.append(row.tolist())
                if tupla in avaliationSets[i]:
                    aset.append(row.tolist())
            k = Knn(tset, j)
            correctPercentage += k.test(aset)     
        generalMean = correctPercentage / len(trainingSets)
        means.append(generalMean)
    matplotlib.pyplot.plot(ks, means)
    matplotlib.pyplot.show()

makeGraph()