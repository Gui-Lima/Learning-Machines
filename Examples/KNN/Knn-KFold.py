import pandas as pd
import sys
from os import path
import matplotlib.pyplot
import numpy as np


sys.path.insert(0, path.abspath('Separating Data/kFoldCrossValidation/'))
sys.path.insert(1, path.abspath('KNN/'))
sys.path.insert(2, path.abspath('DataSets/'))
sys.path.insert(3, path.abspath('./'))

from kCrossValidation import kFoldCrossValidation as kc
from KNN import Knn
import Reading as r
from Global import  knnTypes


#   This is the knn test using kfold cross validation
#   You can run test1 or tes2, representing two different datasets
#   just run test(test1). Run it on the last lines.
#   If you want to run with many k and have a graph, run with graph = True, choose kfold "k" with k = int, and choose knn type with type = weight|adaptative|normal
#  


def test(testId, graph=False, k=5, tp = knnTypes.NORMAL):
    if graph:
        makeGraph(testId['relPath'], testId['columns'], testId['classColumn'], k, tp)
    else:
        simpleKnn(testId['relPath'], testId['columns'], testId['classColumn'], k, tp)


def simpleKnn(relPath, columns, resultColumn,k ,tp):
    dataSet = r.readDataSet(relPath, columns)
    trainingSets = []
    avaliationSets = []
    kfold = kc(dataSet, k, resultColumn, True)
    kfold.run(trainingSets, avaliationSets, stratified = True)
    dataSet = dataSet.apply(pd.to_numeric)

    for i in range(len(trainingSets)):
        tset=[]
        aset=[]
        for index, row in dataSet.iterrows():
            tupla = (dataSet.iloc[index][resultColumn], index)
            if tupla in trainingSets[i]:
                tset.append(row.tolist())
            if tupla in avaliationSets[i]:
                aset.append(row.tolist())
        k = Knn(tset, 1, tp = tp)
        k.test(aset)

def makeGraph(relPath, columns, resultColumn,k ,weight):
    dataSet = r.readDataSet(relPath, columns)
    trainingSets = []
    avaliationSets = []
    kfold = kc(dataSet, k, resultColumn, True)
    kfold.run(trainingSets, avaliationSets, stratified = True)
    dataSet = dataSet.apply(pd.to_numeric)

    ks = [1,2,3,5,7,9,11,13,15]
    means = []
    for j in ks:
        print("Using k = " + str(j))
        correctPercentage = 0
        for i in range(len(trainingSets)):
            tset=[]
            aset=[]
            for index, row in dataSet.iterrows():
                tupla = (dataSet.iloc[index][resultColumn], index)
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






test1 = {'columns' : ['PERCENT_PUB_DATA numeric'
,'ACCESS_TO_PUB_DATA numeric'
,'COUPLING_BETWEEN_OBJECTS numeric'
,'DEPTH numeric'
,'LACK_OF_COHESION_OF_METHODS numeric'
,'NUM_OF_CHILDREN numeric'
,'DEP_ON_CHILD numeric'
,'FAN_IN numeric'
,'RESPONSE_FOR_CLASS numeric'
,'WEIGHTED_METHODS_PER_CLASS numeric'
,'minLOC_BLANK numeric'
,'minBRANCH_COUNT numeric'
,'minLOC_CODE_AND_COMMENT numeric'
,'minLOC_COMMENTS numeric'
,'minCYCLOMATIC_COMPLEXITY numeric'
,'minDESIGN_COMPLEXITY numeric'
,'minESSENTIAL_COMPLEXITY numeric'
,'minLOC_EXECUTABLE numeric'
,'minHALSTEAD_CONTENT numeric'
,'minHALSTEAD_DIFFICULTY numeric'
,'minHALSTEAD_EFFORT numeric'
,'minHALSTEAD_ERROR_EST numeric'
,'minHALSTEAD_LENGTH numeric'
,'minHALSTEAD_LEVEL numeric'
,'minHALSTEAD_PROG_TIME numeric'
,'minHALSTEAD_VOLUME numeric'
,'minNUM_OPERANDS numeric'
,'minNUM_OPERATORS numeric'
,'minNUM_UNIQUE_OPERANDS numeric'
,'minNUM_UNIQUE_OPERATORS numeric'
,'minLOC_TOTAL numeric'
,'maxLOC_BLANK numeric'
,'maxBRANCH_COUNT numeric'
,'maxLOC_CODE_AND_COMMENT numeric'
,'maxLOC_COMMENTS numeric'
,'maxCYCLOMATIC_COMPLEXITY numeric'
,'maxDESIGN_COMPLEXITY numeric'
,'maxESSENTIAL_COMPLEXITY numeric'
,'maxLOC_EXECUTABLE numeric'
,'maxHALSTEAD_CONTENT numeric'
,'maxHALSTEAD_DIFFICULTY numeric'
,'maxHALSTEAD_EFFORT numeric'
,'maxHALSTEAD_ERROR_EST numeric'
,'maxHALSTEAD_LENGTH numeric'
,'maxHALSTEAD_LEVEL numeric'
,'maxHALSTEAD_PROG_TIME numeric'
,'maxHALSTEAD_VOLUME numeric'
,'maxNUM_OPERANDS numeric'
,'maxNUM_OPERATORS numeric'
,'maxNUM_UNIQUE_OPERANDS numeric'
,'maxNUM_UNIQUE_OPERATORS numeric'
,'maxLOC_TOTAL numeric'
,'avgLOC_BLANK numeric'
,'avgBRANCH_COUNT numeric'
,'avgLOC_CODE_AND_COMMENT numeric'
,'avgLOC_COMMENTS numeric'
,'avgCYCLOMATIC_COMPLEXITY numeric'
,'avgDESIGN_COMPLEXITY numeric'
,'avgESSENTIAL_COMPLEXITY numeric'
,'avgLOC_EXECUTABLE numeric'
,'avgHALSTEAD_CONTENT numeric'
,'avgHALSTEAD_DIFFICULTY numeric'
,'avgHALSTEAD_EFFORT numeric'
,'avgHALSTEAD_ERROR_EST numeric'
,'avgHALSTEAD_LENGTH numeric'
,'avgHALSTEAD_LEVEL numeric'
,'avgHALSTEAD_PROG_TIME numeric'
,'avgHALSTEAD_VOLUME numeric'
,'avgNUM_OPERANDS numeric'
,'avgNUM_OPERATORS numeric'
,'avgNUM_UNIQUE_OPERANDS numeric'
,'avgNUM_UNIQUE_OPERATORS numeric'
,'avgLOC_TOTAL numeric'
,'sumLOC_BLANK numeric'
,'sumBRANCH_COUNT numeric'
,'sumLOC_CODE_AND_COMMENT numeric'
,'sumLOC_COMMENTS numeric'
,'sumCYCLOMATIC_COMPLEXITY numeric'
,'sumDESIGN_COMPLEXITY numeric'
,'sumESSENTIAL_COMPLEXITY numeric'
,'sumLOC_EXECUTABLE numeric'
,'sumHALSTEAD_CONTENT numeric'
,'sumHALSTEAD_DIFFICULTY numeric'
,'sumHALSTEAD_EFFORT numeric'
,'sumHALSTEAD_ERROR_EST numeric'
,'sumHALSTEAD_LENGTH numeric'
,'sumHALSTEAD_LEVEL numeric'
,'sumHALSTEAD_PROG_TIME numeric'
,'sumHALSTEAD_VOLUME numeric'
,'sumNUM_OPERANDS numeric'
,'sumNUM_OPERATORS numeric'
,'sumNUM_UNIQUE_OPERANDS numeric'
,'sumNUM_UNIQUE_OPERATORS numeric'
,'sumLOC_TOTAL numeric'
,'DL'], 'relPath' : path.abspath('DataSets/Class-level data for KC1 (Defective or Not) - Software defect prediction/Data.txt'), 'classColumn' : 'DL'}


test2 = {'columns' : ['loc', 'v(g)', 'ev(g)', 'iv(g)', 'n', 'v', 'l', 'd', 'i', 'e','b','t','lOCode', 'lOComment', 'lOBlank', 'lOCodeAndComment', 'uniq_Op', 'uniq_Opnd', 'total_Op', 'total_Opnd', 'branchCount', 'problems']
, 'relPath' : path.abspath('DataSets/KC1 - Software defect prediction/Data.txt')
, 'classColumn' : 'problems'}

