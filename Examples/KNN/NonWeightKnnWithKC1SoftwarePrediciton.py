import pandas as pd
import sys
from os import path

sys.path.insert(0, path.abspath('Separating Data/kFoldCrossValidation/'))
sys.path.insert(1, path.abspath('DataSets/KC1 - Software defect prediction/'))
sys.path.insert(2, path.abspath('KNN/'))
sys.path.insert(3, path.abspath('DataSets/'))

from kCrossValidation import kCrossValidation as kc
import ReadingIt as rd
from KNN import Knn
import Reading as r


dataSet = r.readDataSet(rd.relPath, rd.columns)
groups = dataSet['problems'].tolist()
trainingSets = []
avaliationSets = []
kc(10, groups, trainingSets, avaliationSets)
dataSet = dataSet.apply(pd.to_numeric)

for i in range(len(trainingSets)):
    tset=[]
    aset=[]
    for index, row in dataSet.iterrows():
        tupla = (dataSet.iloc[index]['problems'], index)
        if tupla in trainingSets[i]:
            tset.append(row.tolist())
        if tupla in avaliationSets[i]:
            aset.append(row.tolist())
    k = Knn(tset, 5)
    k.test(aset)
