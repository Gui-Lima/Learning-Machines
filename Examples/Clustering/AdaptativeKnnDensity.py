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
    for index, row in dataSet.iterrows():
        ds.append(row.tolist())
    t = AdaptativeClustering(ds)
    result = t.run()

test3 = {'columns' : ['x', 'y', 'label'],
'relPath' : path.abspath('DataSets/FinlandJoensuuSpiral/Data.txt')
, 'classColumn' : 'label'}

test(test3)
