import pandas as pd
import sys
from os import path
import matplotlib.pyplot
import numpy as np


sys.path.insert(0, path.abspath('Separating Data/kFoldCrossValidation/'))
sys.path.insert(1, path.abspath('Decision Trees/'))
sys.path.insert(2, path.abspath('DataSets/'))
sys.path.insert(3, path.abspath('./'))

from kCrossValidation import kFoldCrossValidation as kc
from ID3 import ID3
import Reading as r
from Global import  knnTypes


def test(testId):
    simpleKnn(testId['relPath'], testId['columns'], testId['classColumn'])


def simpleKnn(relPath, columns, resultColumn):
    dataSet = r.readDataSet(relPath, columns)
    trainingSets = []
    avaliationSets = []
    kfold = kc(dataSet, 5, resultColumn, True)
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
        i = ID3(tset, resultColumn)
        i.printTree()

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



test(test1)