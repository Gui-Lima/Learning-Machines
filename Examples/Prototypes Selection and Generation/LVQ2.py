import pandas as pd
import sys
from os import path
import numpy as np

sys.path.insert(0, path.abspath('Separating Data/kFoldCrossValidation/'))
sys.path.insert(1, path.abspath('Prototype Selecion and Generation/'))
sys.path.insert(2, path.abspath('DataSets/'))
sys.path.insert(3, path.abspath('./'))
sys.path.insert(4, path.abspath('KNN/'))

from kCrossValidation import kFoldCrossValidation as kc
from LVQ21 import LVQ21
from LVQ1 import LVQ1
import Reading as r
from KNN import Knn


def simpleGen(relPath, columns, resultColumn):
    dataSet = r.readDataSet(relPath, columns)
    trainingSets = []
    avaliationSets = []
    kfold = kc(dataSet, 2, resultColumn, True)
    kfold.run(trainingSets, avaliationSets, stratified = True)
    dataSet = dataSet.apply(pd.to_numeric)
    tset=[]
    aset=[]
    for index, row in dataSet.iterrows():
        tupla = (dataSet.iloc[index][resultColumn], index)
        if tupla in trainingSets[0]:
            tset.append(row.tolist())
        if tupla in avaliationSets[0]:
            aset.append(row.tolist())

    q1= LVQ1(tset, resultColumn)
    q2 = LVQ21(tset, resultColumn)
    kn = Knn(tset, 3, 0)
    print("Normal knn")
    kn.test(aset)

    print("LVQ1")
    tset = q1.run()
    kn = Knn(tset, 3, 0)
    kn.test(aset)

    print("After LVQ2")
    tset = q2.run()
    kn = Knn(tset, 3, 0)
    kn.test(aset)

test2 = {'columns' : ['loc', 'v(g)', 'ev(g)', 'iv(g)', 'n', 'v', 'l', 'd', 'i', 'e','b','t','lOCode', 'lOComment', 'lOBlank', 'lOCodeAndComment', 'uniq_Op', 'uniq_Opnd', 'total_Op', 'total_Opnd', 'branchCount', 'problems']
, 'relPath' : path.abspath('DataSets/KC1 - Software defect prediction/Data.txt')
, 'classColumn' : 'problems'}


simpleGen(test2['relPath'], test2['columns'], test2['classColumn'])
