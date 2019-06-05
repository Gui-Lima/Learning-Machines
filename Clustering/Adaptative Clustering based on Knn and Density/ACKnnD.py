import math
import statistics
import sys
from os import path
sys.path.insert(0, path.abspath('KDT/'))
sys.path.insert(1, path.abspath('./'))
sys.path.insert(2, path.abspath('KNN/Distances/'))
sys.path.insert(3, path.abspath('KNN/'))
import Euclidean as euc
from KDT import Kdt
from KNN import Knn

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
        return statistics.mean(l[:_10percent])

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
        f = open(path.abspath("Clustering/Adaptative Clustering based on Knn and Density/Debug/SNN.txt"), "w")
        f.write("KNN of p1 is ")
        f.write('\n')
        f.write(str(KNN[p1]))
        f.write('\n')
        f.write("KNN of p2 is")
        f.write('\n')
        f.write(str(KNN[p2]))
        f.write('\n')
        SNN = [value for value in KNN[p1] if value in KNN[p2]]
        f.write("So SNN is")
        f.write(str(SNN))
        f.close()
        return SNN

    def number_of_samples(self, cluster, clusters):
        i = 0
        for c in clusters.values():
            if c == cluster:
                i += 1
        return i

    def high_end(self, listosa):
        if len(listosa)<2:
            return listosa[0]
        std = statistics.stdev(listosa)
        for i in range(len(listosa)-1):
            if i < len(listosa)/2:
                continue
            else:
                if listosa[i] + std*0.70 < listosa[i+1]:
                    return listosa[i]
                else:
                    continue
        return listosa[-1]

    def low_end(self, listosa):
        if len(listosa)<2:
            return listosa[0]
        std = statistics.stdev(listosa)
        for i in reversed(range(len(listosa))):
            if i > len(listosa)/2:
                continue
            else:
                if listosa[i] - std*0.70 > listosa[i-1]:
                    return listosa[i]
                else:
                    continue
        return listosa[0]

    def euclidean_dist(self, p1, p2):
        if len(p1) != len(p2):
            raise Exception("Length of two numbers in euclidean distance must be equal")
        else:
            aux = []
            for i in range(len(p1)):
                temp = (p1[i]-p2[i])**2
                aux.append(temp)
            total = sum(aux)
        return math.sqrt(total)
    
    def get_nearest_neighbors(self, instance, data, k):
        print(instance)
        print(data[0])
        calc_dist = lambda x: self.euclidean_dist(instance[1], x[1])
        data.sort(key=calc_dist)
        return data[:k]

    def run(self):
        Result = {}
        #  STEP 1
        nExamples = len(self.data)
        nAtributes = len(self.data[0])
        alpha = nExamples/self.getAlpha(nAtributes)
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
        wq = []
        for a in range(len(self.data)):
            wq.append((a, self.data[a]))
        f = open(path.abspath("Clustering/Adaptative Clustering based on Knn and Density/Debug/d5NN.txt"), "w")
        for i in range(nExamples):
            kNN_index.append(self.get_nearest_neighbors((i, self.data[i]), wq, k))
            f.write("Closest elements to " + str(i) + " are ")
            f.write('\n')
            temp = [Distance[i][kNN_index[i][1][0]][1], Distance[i][kNN_index[i][2][0]][1],Distance[i][kNN_index[i][3][0]][1],Distance[i][kNN_index[i][4][0]][1],Distance[i][kNN_index[i][5][0]][1]]
            f.write(str( [ Distance[i][kNN_index[i][1][0]], Distance[i][kNN_index[i][2][0]], Distance[i][kNN_index[i][3][0]],Distance[i][kNN_index[i][4][0]],Distance[i][kNN_index[i][5][0]]]))
            f.write('\n')
            d5NN.append(statistics.mean(temp))
            f.write("Resulting a mean of ")
            f.write(str(d5NN[i]))
            f.write('\n')
            f.write("----------------")
        f.close()

        # STEP 4
        d5NN = sorted(d5NN)
        R = self.high_end(d5NN)
        print("Global R: " + str(R))

        # STEP 5
        f = open(path.abspath("Clustering/Adaptative Clustering based on Knn and Density/Debug/SR.txt"), "w")
        Rho = {}
        for i in range(nExamples):
            surronding_region = list(filter(lambda x: x[1] < R, Distance[i]))
            f.write("SR of element " + str(i) + " is ")
            f.write('\n')
            f.write(str(surronding_region))
            indices = [i[0] for i in surronding_region]
            values =  [i[1] for i in surronding_region]
            std = 0 if len(values) < 2 else  statistics.stdev(values)
            mean = statistics.mean(values)
            surronding_region = sorted(surronding_region, key=lambda x: x[1])
            values = sorted(values)
            d1 = self.get10percentClosestMean(values)
            d2 = self.low_end(values)
            a = (d1 + d2)/2
            b = min(a + 2 * std, R)
            Rho[i] = 0
            for j in surronding_region:
                Rho[i] += self.fab(self.data[i], self.data[j[0]], a ,b)
            f.write('\n')
            f.write(" and as so, density of " + str(i) + " is " + str(Rho[i]))
            f.write('\n')
            f.write('-----------')
        f.close()

        # STEP 6
        Arr = []
        for i in range(nExamples):
            Arr.append((Rho[i], i))
        Arr = sorted(Arr, key=lambda x: x[0], reverse=True)
        sortedRhos = [i[0] for i in Arr]
        T = self.lowerQuartile(sortedRhos)
        print("Threshold: " + str(T))

        # STEP 7
        f = open(path.abspath("Clustering/Adaptative Clustering based on Knn and Density/Debug/Clusters.txt"), "w")
        Order_of_Cluster = 0
        Tag = [False] * nExamples
        Queue = []
        for i in range(nExamples):
            f.write("Element " + str(Arr[i][1]))
            if Tag[Arr[i][1]]:
                f.write("is alredy in a cluster!")
                f.write('\n')
                continue
            if Arr[i][0] == 0:
                f.write("is a outlier")
                Tag[Arr[i][1]] = True
                f.write('\n')
                continue
            if Arr[i][0] < T:
                f.write(" is below treshold " + str(T) +" because is " + str(Arr[i][0]))
                f.write('\n')
                continue
            f.write('\n')
            f.write("alright, is a new cluster")
            Order_of_Cluster += 1
            Queue = []
            Queue.append(Arr[i][1])
            f.write("Creating this new Queue w " + str(Arr[i][1]))
            while Queue:
                f.write("\n")
                first = Queue.pop(0)
                Tag[first] = True
                f.write("Queue element : " + str(first))
                surronding_region = filter(lambda x: x[1] < R, Distance[first])
                f.write('\n')
                surronding_regionTreshold = sorted(surronding_region, key=lambda x: x[1])[-1][1]
                f.write("Surronding Region Treshold: " + str(surronding_regionTreshold))
                temp = []
                for p in range(len(kNN_index[first])):
                    temp.append(Distance[first][kNN_index[first][p][0]])
                elementsInKnnAndSR = list(filter(lambda x: x[1] < surronding_regionTreshold, temp))
                f.write('\n')
                f.write("Elements in Knn and SR of " + str(first))
                f.write(str(elementsInKnnAndSR))
                elementsInKnnAndSRDensity = []
                for element in elementsInKnnAndSR:
                    elementsInKnnAndSRDensity.append(Rho[element[0]])
                aver = statistics.mean(elementsInKnnAndSRDensity)
                f.write('\n')
                f.write("Aver of this elements density: " + str(aver))
                LocalT = aver * beta
                for e in range(k):
                    f.write('\n')
                    element = kNN_index[first][e]
                    f.write('Trying to add ' + str(element) + " to cluster")
                    if euc.getDistance(element[1],self.data[first]) > R:
                        f.write(" but distance is too high! because it is " + str(euc.getDistance(element[1], self.data[first])))
                        continue
                    co_NN = len(self.getSharedNeighbors(element[0], first, kNN_index))
                    f.write(" ssn of " + str(element[0]) + " with " + str(first) + " is " + str(co_NN))
                    if Rho[element[0]]:
                        f.write(" ... appeding sucefull, this element cluster is " + str(Order_of_Cluster))
                        Result[element[0]] = Order_of_Cluster
                        if Rho[element[0]] >= LocalT and co_NN > k * gama and not Tag[element[0]] and element[0] not in Queue:
                            Queue.append(element[0])
                            f.write("... wow, this will be added to Queue, since it's density is " + str(Rho[element[0]]) + " and the Treshold is " + str(LocalT))

                f.write('\n')
        for cluster in range(1, Order_of_Cluster):
            if self.number_of_samples(cluster, Result) <= 5:
                for j in range(nExamples): 
                    if j in Result and Result[j] == cluster:
                        Result[j] = 0
        self.nClusters = Order_of_Cluster
        print(Result)
        return Result



 