import math
import statistics
import sys
from os import path
sys.path.insert(0, path.abspath('KDT/'))
sys.path.insert(1, path.abspath('./'))
sys.path.insert(2, path.abspath('KNN/Distances/'))
import Euclidean as euc
from KDT import Kdt

class AdaptativeClustering:

    def __init__(self, data):
        self.data = data
    
    def getNumberOfClusters(self):
        return self.nClusters

    def maxValueInColumn(self, columnIndex):
        max = self.data[0][columnIndex]
        for i in self.data:
            if max < i[columnIndex]:
                max = i[columnIndex]
        return max

    def minValueInColumn(self, columnIndex):
        min = self.data[0][columnIndex]
        for i in self.data:
            if min > i[columnIndex]:
                min = i[columnIndex]
        return min
    
    def getNeighbors(self, point):
        distances = []
        for i in range(len(self.data)):
            distances.append( (i, euc.getDistance(self.data[i], point)))
        return distances

    def getAllDistances(self):
        distances = {}
        for i in range(len(self.data)):
            distances[i] = self.getNeighbors(self.data[i])
        return distances

    def get10percentClosestMean(self, l):
        _10percent = math.ceil(0.1 * len(l)) 
        return statistics.mean(l[_10percent:])

    def fab(self, i, j, a, b):
        d = euc.getDistance(i, j)
        if 0<= d and d <= a:
            return 1
        if d > b:
            return 0
        else:
            return (b - d)/(b-a)

    def lowerQuartile(self, l):
        half = int(len(l)/2)
        return statistics.median(l[:half])

    def getAlpha(self, nAtributes):
        alpha = 0
        for i in range(nAtributes):
            max_s = self.maxValueInColumn(i)
            min_s = self.minValueInColumn(i)
            rs = max_s - min_s
            alpha += rs
        return alpha
    
    def getSharedNeighbors(self, p1, p2, KNN):
        SNN = [value for value in KNN[p1] if value in KNN[p2]] 
        return SNN

    def number_of_samples(self, cluster, clusters):
        i = 0
        for c in clusters:
            if c == cluster:
                i += 1
        return i

    def run(self):
        Result = {}
        #  STEP 1
        nExamples = len(self.data)
        nAtributes = len(self.data[0])
        alpha = self.getAlpha(nAtributes)
        beta = ( 3.5 * alpha + 45 ) / 100
        gama = ( 2   * alpha + 18 ) / 100
        k = max(10, nExamples/100)
        print("Calculated the parameters:")
        print("N: " + str(nExamples))
        print("m: " + str(nAtributes))
        print("Alpha: " + str(alpha))
        print("beta: " + str(beta))
        print("gama: " + str(gama))
        print("k: " + str(k))

        # STEP 2
        KDTree = Kdt(self.data)
        KDTreeModel = KDTree.run()

        # STEP 3
        kNN_index = []
        Distance = self.getAllDistances()
        d5NN = []
        for i in range(nExamples):
            kNN_index.append(KDTree.find_neighbors(KDTreeModel, self.data[i], k))
            print("Closest neighbors of " + str(i) + " are ")
            print(kNN_index[i])
            print("Distances:")
            temp = [Distance[i][kNN_index[i][0][0]][1], Distance[i][kNN_index[i][1][0]][1],Distance[i][kNN_index[i][2][0]][1],Distance[i][kNN_index[i][3][0]][1],Distance[i][kNN_index[i][4][0]][1]]
            print(temp)
            d5NN.append(statistics.mean(temp))

        # STEP 4
        d5NN = sorted(d5NN)
        R = d5NN[nExamples -1 ]
        print("Global R: " + str(R))

        # STEP 5
        Rho = {}
        for i in range(nExamples):
            surronding_region = list(filter(lambda x: x[1] < R, Distance[i]))
            print("SR of " + str(i))
            print(surronding_region)
            indices = [i[0] for i in surronding_region]
            values =  [i[1] for i in surronding_region]
            mean = statistics.mean(values)
            std  = statistics.stdev(values)
            surronding_region = sorted(surronding_region, key=lambda x: x[1])
            values = sorted(values)
            d1 = self.get10percentClosestMean(values)
            d2 = surronding_region[0][1]
            a = (d1 + d2)/2
            b = min(a + 2 * std, R)
            for j in surronding_region:
                Rho[i] = self.fab(self.data[i], self.data[j[0]], a ,b)
            print("-")

        # STEP 6
        Arr = []
        for i in range(nExamples):
            Arr.append((Rho[i], i))
        Arr = sorted(Arr, key=lambda x: x[0])
        sortedRhos = [i[0] for i in Arr]
        T = self.lowerQuartile(sortedRhos)

        # STEP 7
        Order_of_Cluster = 0
        Tag = [False] * nExamples
        Queue = []
        for i in range(nExamples):
            if Tag[Arr[i][1]]:
                continue
            if Arr[i][0] == 0:
                Tag[Arr[i][1]] = True
                continue
            if Arr[i][0] < T:
                continue
            Order_of_Cluster += 1
            Tag[Arr[i][1]] = True
            Queue = []
            Queue.append(Arr[i][1])
            while Queue:
                first = Queue.pop(0)
                surronding_region = filter(lambda x: x[1] < R, Distance[i])
                surronding_regionTreshold = sorted(surronding_region, key=lambda x: x[1])[-1][1]
                temp = []
                for p in range(len(kNN_index[first])):
                    temp.append(Distance[first][kNN_index[first][p][0]])
                print(temp)
                elementsInKnnAndSR = filter(lambda x: x[1] < surronding_regionTreshold, temp)
                elementsInKnnAndSRDensity = []
                for element in elementsInKnnAndSR:
                    elementsInKnnAndSRDensity.append(Rho[element[0]])
                aver = statistics.mean(elementsInKnnAndSRDensity)
                LocalT = aver * beta
                for e in range(k):
                    element = kNN_index[first][e]
                    if euc.getDistance(element[1],self.data[first]) > R:
                        continue
                    co_NN = len(self.getSharedNeighbors(element[0], first, kNN_index))
                    Tag[element[0]] = True
                    if Rho[element[0]]:
                        Result[element[0]] = Order_of_Cluster
                    if Rho[element[0]] >= LocalT and co_NN > k * gama:
                        Queue.push(element)
        for cluster in range(1, Order_of_Cluster):
            if self.number_of_samples(cluster, Result) <= 5:
                for j in range(nExamples): 
                    if j in Result and Result[j] == cluster:
                        Result[j] = 0
        self.nClusters = Order_of_Cluster
        print(Result)
        return Result



 