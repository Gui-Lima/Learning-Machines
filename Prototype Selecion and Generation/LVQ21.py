import random
from os import path
import sys
sys.path.insert(0, path.abspath('KNN/Distances/'))
sys.path.insert(1, path.abspath('./'))  
sys.path.insert(2, path.abspath('Prototype Selecion and Generation/'))
import Euclidean as euc
from Global import listOps
from LVQ1 import LVQ1

class LVQ21:

    def __init__(self, trainingSet, classColumn, w=0.66, newSet= [], seed= None):
        self.trainingSet = trainingSet
        self.newSet = newSet
        self.classColumn = classColumn
        self.groups = [i[-1] for i in trainingSet]
        self.seed = seed
        self.w = w

    def isInWindow(self, di, dj):
        didj = di/dj
        djdi = dj/di
        s = (1 - self.w) / (1 + self.w)
        return min(didj, djdi) > s

    def getRandomElement(self, inplace=False):
        choice = random.choice(self.trainingSet)
        if inplace:
            self.trainingSet.remove(choice)
        return choice

    def getRandomElementFromClass(self, requestedClass, inplace=False):
        choice = random.choice(self.trainingSet)
        while self.getClass(choice) != requestedClass:
            choice = random.choice(self.trainingSet)
        if inplace:
            self.trainingSet.remove(choice)
        return choice

    def getClass(self, point):
        return point[-1:][0]


    def getNearestNeighbors(self, point, dataset, inplace=False):
        distances = []
        for i in range(len(dataset)):
            distances.append((dataset[i] , euc.getDistance(dataset[i], point)))
        sortedDistances = sorted(distances, key= lambda  tup: tup[1])
        element1 = sortedDistances[0]
        element2 = sortedDistances[1]
        if inplace:
            dataset.remove(element1[0])
            dataset.remove(element2[0])
        return (element1, element2)

    def alpha(self, alpha, s):
        return alpha/(1 + s * alpha)

    def run(self):
        print("Starting LVQ2.1")
        lo = listOps()
        lvq = LVQ1(self.trainingSet, self.classColumn)
        startingElements = lvq.run()
        self.writeElementsInFile(startingElements)
        nIter = 0
        nIterMax = 100
        Alpha = 0.99
        while nIter < nIterMax:
            randomElement = self.getRandomElement(inplace=True)
            nearestNeighbors = self.getNearestNeighbors(randomElement, startingElements, inplace=True)
            neighbor1 = nearestNeighbors[0][0]
            neighbor2 = nearestNeighbors[1][0]
            distance1 = nearestNeighbors[0][1]
            distance2 = nearestNeighbors[1][1]           

            if randomElement is neighbor2 or randomElement is neighbor1:
                break
            randomElementClass = self.getClass(randomElement)
            nearestNeighborClass1 = self.getClass(neighbor1)
            nearestNeighborClass2 = self.getClass(neighbor2)
            randomElement.pop()
            neighbor1.pop()
            neighbor2.pop()
            if self.isInWindow(distance1, distance2) and not (nearestNeighborClass1 is nearestNeighborClass2) and ((nearestNeighborClass1 is randomElementClass) or (nearestNeighborClass2 is randomElementClass)):
                if nearestNeighborClass1 is randomElementClass:
                    neighbor1 = lo.listSum(neighbor1, lo.scalarMultList(Alpha, lo.listSub(randomElement, neighbor1)))
                    neighbor2 = lo.listSub(neighbor2, lo.scalarMultList(Alpha, lo.listSub(randomElement, neighbor2)))
                    Alpha = self.alpha(Alpha, 1)
                if nearestNeighborClass2 is randomElementClass:
                    neighbor1 = lo.listSub(neighbor1, lo.scalarMultList(Alpha, lo.listSub(randomElement, neighbor1)))
                    neighbor2 = lo.listSum(neighbor2, lo.scalarMultList(Alpha, lo.listSub(randomElement, neighbor2)))
                    Alpha = self.alpha(Alpha, -1)        

            neighbor1.append(nearestNeighborClass1)
            neighbor2.append(nearestNeighborClass2)
            randomElement.append(randomElementClass)
            startingElements.append(neighbor1)
            startingElements.append(neighbor2)
            nIter += 1
        self.writeNewElementsInFile(startingElements)
        return startingElements

    def writeElementsInFile(self, elements):
        f = open(path.abspath("Prototype Selecion and Generation/ElementsDebug.txt"), "w")
        f.write("Initial Prototypes")
        f.write("\n")
        for i in elements:
            for j in i:
                f.write(str(j))
                f.write(", ")
            f.write("\n")
        f.close()

    def writeNewElementsInFile(self, elements):
        f = open(path.abspath("Prototype Selecion and Generation/ElementsDebug.txt"), "a")
        f.write("New Prototypes")
        f.write("\n")
        for i in elements:
            for j in i:
                f.write(str(j))
                f.write(", ")
            f.write("\n")
        f.close()
