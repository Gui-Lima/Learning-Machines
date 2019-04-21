import random
from os import path
import sys
sys.path.insert(0, path.abspath('KNN/Distances/'))
sys.path.insert(1, path.abspath('./'))
sys.path.insert(2, path.abspath('Prototype Selecion and Generation/'))
import Euclidean as euc
from Global import listOps
from InitialSelections import Selection as sel

class LVQ1:

    def __init__(self, trainingSet, classColumn, newSet = [],seed=None):
        self.trainingSet = trainingSet
        self.newSet = newSet
        self.classColumn = classColumn
        self.groups = [i[-1] for i in trainingSet]
        self.seed = seed

    def selection(self, prototypes, inplace=False):
        s = sel(self.trainingSet, self.classColumn, prototypes)
        elements = s.randomNSelectionEachClass()
        return elements

    def getRandomElement(self):
        choice = random.choice(self.trainingSet)
        return choice

    def getNearestNeighbor(self, point, dataset, inplace=False):
        distances = []
        for i in range(len(dataset)):
            distances.append((dataset[i] , euc.getDistance(dataset[i], point)))
        sortedDistances = sorted(distances, key= lambda  tup: tup[1])
        element = sortedDistances[:1][0][0]
        if inplace:
            dataset.remove(element)
        return element

    def getClass(self, point):
        return point[-1:][0]

    def alpha(self, alpha, s):
        return alpha/(1 + s*alpha)

    def run(self, nPrototypes=5):
        print("Starting LVQ1")
        lo = listOps()
        startingElements = self.selection(nPrototypes)
        self.writeElementsInFile(startingElements)
        nIter = 0
        nIterMax = 100
        Alpha = 0.99
        while nIter < nIterMax:
            randomElement = self.getRandomElement()
            nearestNeighbor = self.getNearestNeighbor(randomElement, startingElements)
            startingElements.remove(nearestNeighbor)
            if randomElement is nearestNeighbor:
                break
            randomElementClass = self.getClass(randomElement)
            nearestNeighborClass = self.getClass(nearestNeighbor)
            randomElement.pop()
            nearestNeighbor.pop()
            if randomElementClass is nearestNeighborClass:
                nearestNeighbor = lo.listSub(nearestNeighbor, (lo.scalarMultList(Alpha, lo.listSub(randomElement, nearestNeighbor))))
                Alpha = self.alpha(Alpha, 1)
            else:
                nearestNeighbor = lo.listSum(nearestNeighbor, (lo.scalarMultList(Alpha, lo.listSub(randomElement, nearestNeighbor))))
                Alpha = self.alpha(Alpha, -1)
            nearestNeighbor.append(nearestNeighborClass)
            randomElement.append(randomElementClass)
            startingElements.append(nearestNeighbor)
            nIter += 1
        self.writeNewElementsInFile(startingElements)
        return startingElements

    # Debug Functions
    
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