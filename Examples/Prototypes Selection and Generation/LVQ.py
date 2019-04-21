#   This is a test file, it's useful to see how to run the machines. To see the implementations, you should go to the folder 'Prototype Selecion and Generation'
#
#

import pandas as pd
import sys
from os import path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, path.abspath('Separating Data/kFoldCrossValidation/'))
sys.path.insert(1, path.abspath('Prototype Selecion and Generation/'))
sys.path.insert(2, path.abspath('DataSets/'))
sys.path.insert(3, path.abspath('./'))
sys.path.insert(4, path.abspath('KNN/'))
sys.path.insert(5, path.abspath('Examples/'))

from kCrossValidation import kFoldCrossValidation as kc
from LVQ1 import LVQ1
from LVQ21 import LVQ21
from LVQ3 import LVQ3
import Reading as r
from KNN import Knn
from TestsSets import test1, test2

# @TEST
def _LVQ1(relPath, columns, resultColumn):
    dataSet = r.readDataSet(relPath, columns)
    trainingSets = []
    avaliationSets = []
    kfold = kc(dataSet, 10, resultColumn, True)
    kfold.run(trainingSets, avaliationSets, stratified = True)
    dataSet = dataSet.apply(pd.to_numeric)
    tset=[]
    aset=[]
    for i in range(len(trainingSets)):
        print("\n")
        print(" --------- FOLD " + str(i+1) +  " ----------------")
        tset=[]
        aset=[]
        for index, row in dataSet.iterrows():
            tupla = (dataSet.iloc[index][resultColumn], index)
            if tupla in trainingSets[i]:
                tset.append(row.tolist())
            if tupla in avaliationSets[i]:
                aset.append(row.tolist())
        print("------------- SIMPLE KNN ----------------")
        k = Knn(tset, 3)
        k.test(aset)
        lvq = LVQ1(tset, resultColumn)
        newtset = lvq.run()
        print("-------------- LVQ1 ---------------------")
        k = Knn(newtset, 3)
        k.test(aset)

# @TEST
def _LVQ2(relPath, columns, resultColumn):
    dataSet = r.readDataSet(relPath, columns)
    trainingSets = []
    avaliationSets = []
    kfold = kc(dataSet, 10, resultColumn, True)
    kfold.run(trainingSets, avaliationSets, stratified = True)
    dataSet = dataSet.apply(pd.to_numeric)
    tset=[]
    aset=[]
    for i in range(len(trainingSets)):
        print("\n")
        print(" --------- FOLD " + str(i+1) +  " ----------------")
        tset=[]
        aset=[]
        for index, row in dataSet.iterrows():
            tupla = (dataSet.iloc[index][resultColumn], index)
            if tupla in trainingSets[i]:
                tset.append(row.tolist())
            if tupla in avaliationSets[i]:
                aset.append(row.tolist())
        print("------------- SIMPLE KNN ----------------")
        k = Knn(tset, 3)
        k.test(aset)
        lvq = LVQ21(tset, resultColumn)
        newtset = lvq.run()
        print("-------------- LVQ2.1 ---------------------")
        k = Knn(newtset, 3)
        k.test(aset)


# @TEST
def _LVQ3(relPath, columns, resultColumn):
    dataSet = r.readDataSet(relPath, columns)
    trainingSets = []
    avaliationSets = []
    kfold = kc(dataSet, 10, resultColumn, True)
    kfold.run(trainingSets, avaliationSets, stratified = True)
    dataSet = dataSet.apply(pd.to_numeric)
    tset=[]
    aset=[]
    for i in range(len(trainingSets)):
        print("\n")
        print(" --------- FOLD " + str(i+1) +  " ----------------")
        tset=[]
        aset=[]
        for index, row in dataSet.iterrows():
            tupla = (dataSet.iloc[index][resultColumn], index)
            if tupla in trainingSets[i]:
                tset.append(row.tolist())
            if tupla in avaliationSets[i]:
                aset.append(row.tolist())
        print("------------- SIMPLE KNN ----------------")
        k = Knn(tset, 3)
        k.test(aset)
        lvq = LVQ3(tset, resultColumn)
        newtset = lvq.run()
        print("-------------- LVQ3 ----------------------")
        k = Knn(newtset, 3)
        k.test(aset)

# @TEST
def _makeGraph(relPath, columns, resultColumn):
    dataSet = r.readDataSet(relPath, columns)
    trainingSets = []
    avaliationSets = []
    kfold = kc(dataSet, 10, resultColumn, True)
    kfold.run(trainingSets, avaliationSets, stratified = True)
    dataSet = dataSet.apply(pd.to_numeric)

    ks = [1,3]
    nPrototypes = [3,5,10,20]
    
    for k in ks:
        meansGeral = []
        meansFalse = []
        meansTrue = []
        for j in nPrototypes:
            correctnessPercentage = 0
            correctTrue = 0
            correctFalse = 0
            for i in range(len(trainingSets)):
                print("\n")
                print(" --------- FOLD " + str(i+1) +  " ----------------")
                tset=[]
                aset=[]
                for index, row in dataSet.iterrows():
                    tupla = (dataSet.iloc[index][resultColumn], index)
                    if tupla in trainingSets[i]:
                        tset.append(row.tolist())
                    if tupla in avaliationSets[i]:
                        aset.append(row.tolist())
                lvq = LVQ3(tset, resultColumn)
                newtset = lvq.run(nPrototypes=j)
                kn = Knn(newtset, k)
                result = kn.test(aset)
                correctnessPercentage += result[0]
                classErrors = result[1]
                classNumbers = result[2]
                correctFalse += (classErrors[False]/classNumbers[False]) if False in classErrors.keys() else 0
                correctTrue += (classErrors[True]/classNumbers[True]) if True in classErrors.keys() else 0
            meansGeral.append(correctnessPercentage/len(trainingSets))
            meansFalse.append(correctFalse/len(trainingSets))
            meansTrue.append(correctTrue/len(trainingSets))
        plt.ylim(0,1)    
        plt.plot(nPrototypes, meansGeral, 'r', label='general')
        plt.plot(nPrototypes, meansFalse, 'g', label='false')
        plt.plot(nPrototypes, meansTrue,  'b', label='true')
        plt.legend(loc='upper left')
        plt.show()





_makeGraph(test1['relPath'], test1['columns'], test1['classColumn'])
