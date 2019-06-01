## Done by @ClaudioCarvalhoo
import math
import sys
from os import path
sys.path.insert(2, path.abspath('KNN/Distances/'))
import Euclidean as euc
import statistics

class Kdt:

    def __init__(self, data):
        self.data = data

    def index_data(self):
        data = []
        for i in range(len(self.data)):
            data.append( (i,self.data[i]) )
        return data

    def run(self):
        quant_att = len(self.data[0])
        data = self.index_data()
        self.model = self.kdtree_recursion(data, quant_att, 0)
        return self.model

    def median_column(self, data, itera):
        total = 0
        print(data)
        for i in data:
            total += i[1][itera]
        return float(total)/len(data)

    def kdtree_recursion(self, data, quant_att, itera):
        left = []
        right = []
        mid = self.median_column(data, itera)
        if len(data) == 1:
            return {"end": data[0]}
        for i in data:
            if i[1][itera]<= mid:
                left.append(i)
            else:
                right.append(i)
        if itera == quant_att-1:
            return {mid: {"left": {"end": left} , "right": {"end": right}} }
        elif len(left)!=0 and len(right)!=0:
            tree = {}
            tree[mid] = {"left": self.kdtree_recursion(left, quant_att, itera+1) , "right": self.kdtree_recursion(right, quant_att, itera+1)}
            return tree

    def find_neighbors(self, model, instance, k, itera=0):
        mid = list(model.keys())[0]
        if instance[itera] <= mid:
            model = model[mid]["left"]
        elif instance[itera] > mid:
            model = model[mid]["right"]
        if list(model.keys())[0] == "end":
            return self.knn(model["end"], instance, k)
        else:
            return self.find_neighbors(model, instance,k, itera+1)

    def knn(self, neighbors, instance, k):
        calc_dist = lambda x: euc.getDistance(instance, x[1])
        neighbors.sort(key=calc_dist)
        return neighbors[:k]
        
    def printTree(self):
        for i in self.model:
            print(i)