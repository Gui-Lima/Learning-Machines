import sys
from os import path
sys.path.insert(0, path.abspath('KNN/Distances/'))
sys.path.insert(1, path.abspath('./'))
import Euclidean as euc
from progress.bar import Bar
import math
from TimeMeasure import profile as profile
from TimeMeasure import print_prof_data
from TimeMeasure import clear_prof_data

from Global import  knnTypes


class Knn:

    @profile
    def __init__(self, trainingSet, k, tp = knnTypes.NORMAL):
        self.trainingSet = trainingSet
        self.k = k
        self.tp = tp
        if tp == knnTypes.ADAPTATIVE:
            print("Since you've chosen Adaptative, here's the calculations: ")
            self.r = self.calculateR()
        print_prof_data()


    
    def computeVote(self, neighbors, i):
        result = 0
        if self.tp == knnTypes.WEIGHTED:
            if neighbors[i][1] != 0:
                result = 1/math.pow(neighbors[i][1], 2)
        elif self.tp == knnTypes.ADAPTATIVE:
            if self.r[i] != 0:
                result = neighbors[i][1]/self.r[i]
        elif self.tp == knnTypes.NORMAL:
            result = 1
        return result
            

    def getClass(self, neighbors):
        votes = {}
        for i in range(len(neighbors)):
            response = neighbors[i][0][-1]
            if response in votes:
                votes[response] += self.computeVote(neighbors, i)
            else:
                votes[response] = self.computeVote(neighbors, i)

        return sorted(votes)[:1]


    def calculateR(self):
        bar = Bar('Processing R', max=len(self.trainingSet))
        r = {}
        index = 0
        for i in self.trainingSet:
            r[index] = 0
            iNeighbors = self.getNeighbors(i, useK=False)
            for j in iNeighbors:
                if j[0][-1] == i[-1]:
                    r[index] = j[1]
                else:
                    break
            index += 1
            bar.next()
        bar.finish()
        return r

    def getNeighbors(self, newPoint, useK = True):
        distances = []
        for i in range(len(self.trainingSet)):
            distances.append((self.trainingSet[i] , euc.getDistance(self.trainingSet[i], newPoint)))
        sortedDistances = sorted(distances, key= lambda  tup: tup[1])
        return (sortedDistances[:self.k] if useK else sortedDistances)

    def getNewElementClass(self, newPoint):
        neighbors = self.getNeighbors(newPoint)
        return self.getClass(neighbors)


    def showClassErrors(self, classErrors, classNumbers):
        for i in classErrors.keys():
            print("Acerto na classe: " + str(i) + " " + str(classErrors[i]/classNumbers[i]))

    @profile
    def test(self, avaliationSet):
        bar = Bar('Processing new examples', max=len(avaliationSet))
        corrects = 0
        classErrors = {}
        classNumbers = {}

        for i in range(len(avaliationSet)):
            example = avaliationSet[i]
            actualClass = example[-1]

            if actualClass in classNumbers:
                classNumbers[actualClass] += 1
            else:
                classNumbers[actualClass] = 1

            #print("new example to be avaliated :" + str(example))
            #print("The actual class of this example is : " + str(actualClass))
            predictedClass = self.getNewElementClass(example)
            #print("Predicted class is : " + str(predictedClass[0]))
            if (actualClass == predictedClass[0]):
                if actualClass in classErrors:
                    classErrors[actualClass] += 1
                elif actualClass not in classErrors:
                    classErrors[actualClass] = 1
                corrects += 1
            bar.next()
        bar.finish()
        print("Of " + str(len(avaliationSet)) + " examples, we got " + str(corrects) + " of them right!")
        print("Final percentage correctness :" + str(corrects/len(avaliationSet)))
        self.showClassErrors(classErrors, classNumbers)
        print_prof_data()
        return corrects/len(avaliationSet)

