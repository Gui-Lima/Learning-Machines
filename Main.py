# This is where i test stuff
import pandas as pd
import sys
from os import path
sys.path.insert(0, path.abspath('Separating Data/kFoldCrossValidation/'))
sys.path.insert(1, path.abspath('DataSets/KC1 - Software defect prediction/'))
sys.path.insert(2, path.abspath('KNN/'))

from kCrossValidation import kCrossValidation as kc
from ReadingIt import readDataSet as rd
from KNN import Knn


columns = ['loc', 'v(g)', 'ev(g)', 'iv(g)', 'n', 'v', 'l', 'd', 'i', 'e','b','t','lOCode', 'lOComment', 'lOBlank', 'lOCodeAndComment', 'uniq_Op', 'uniq_Opnd', 'total_Op', 'total_Opnd', 'branchCount', 'problems']
dataSet = rd(path.abspath('DataSets/KC1 - Software defect prediction/Data.txt'), columns)
groups = dataSet['problems'].tolist()
trainingSets = []
avaliationSets = []
kc(5, groups, trainingSets, avaliationSets)


for i in range(len(trainingSets)):
    tset=[]
    aset=[]
    for index, row in dataSet.iterrows():
        tupla = (dataSet.iloc[index]['problems'], index)
        if tupla in trainingSets[i]:
            tset.append(row.tolist())
        if tupla in avaliationSets[i]:
            aset.append(row.tolist())
    k = Knn(tset, 3)
    k.test(aset)
