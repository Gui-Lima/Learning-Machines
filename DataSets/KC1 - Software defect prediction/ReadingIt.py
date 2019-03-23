from os import path
import pandas

def readDataSet(datasetRelativePath, columnNames='none'):
    df = pandas.read_csv(path.abspath(datasetRelativePath), names=columnNames)
    return df


