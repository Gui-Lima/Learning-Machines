import pandas as pd
import sys
from os import path
import matplotlib.pyplot
import numpy as np

sys.path.insert(0, path.abspath('Clustering/Adaptative Clustering based on Knn and Density/'))
sys.path.insert(1, path.abspath('DataSets/'))
sys.path.insert(2, path.abspath('./'))
from ACKnnD import AdaptativeClustering
import Reading as r


def test(testId):
    makeGraph(testId['relPath'], testId['columns'], testId['classColumn'])

def makeGraph(relPath, columns, classColumn):
    dataSet = r.readDataSet(relPath, columns)
    dataSet.drop(dataSet.columns[[-1,]], axis=1, inplace=True)
    ds = []
    color = ['#FF0000', '#0055FF', '#00FF5E', '#008080', '#F7FF00', '#0D0D0B', '#00FFDE', '#6A98D2', '#239C2F', '#9FC9A3', '#6619CA', '#B98B20', '#6D2FA4']
    for index, row in dataSet.iterrows():
        ds.append(row.tolist())
    t = AdaptativeClustering(ds)
    result = t.run()
    xAxis = []
    yAxis = []
    c = 0
    for i in set(result.values()):
        xAxis = []
        yAxis = []
        for j in result.keys():
            if result[j] == i:
                xAxis.append(ds[j][0])
                yAxis.append(ds[j][1])
        matplotlib.pyplot.plot(xAxis, yAxis, 'ro', color=color[c])
        c += 1
    matplotlib.pyplot.show()


test3 = {'columns' : ['x', 'y', 'label'],
'relPath' : path.abspath('DataSets/FinlandJoensuuSpiral/Data.txt')
, 'classColumn' : 'label'}

test4 = {'columns' : ['x', 'y', 'label'],
'relPath' : path.abspath('DataSets/FinlandJoesuuFlame/Data.txt')
, 'classColumn' : 'label'}

test5 = {'columns' : ['x', 'y', 'label'],
'relPath' : path.abspath('DataSets/FinlandJoensuuJain/data.txt')
, 'classColumn' : 'label'}

test6 = {'columns' : ['x', 'y', 'label'],
'relPath' : path.abspath('DataSets/FinlandJoensuuAgregation/Data.txt')
, 'classColumn' : 'label'}

test(test6)
