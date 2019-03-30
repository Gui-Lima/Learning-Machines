import sys
from os import path
sys.path.insert(0, path.abspath('KNN/Distances/'))
sys.path.insert(1, path.abspath('./'))
import Euclidean as euc
from progress.bar import Bar
import math
from TimeMeasure import profile as profile
from TimeMeasure import print_prof_data

class Knn:

    def __init__(self, trainingSet, k, weighted=False):
      self.trainingSet = trainingSet
      self.k = k
      self.weighted = weighted

    def getClass(self, neighbors):
        votes = {}
        for i in range(len(neighbors)):
            response = neighbors[i][0][-1]
            if response in votes:
                if self.weighted:
                    if neighbors[i][1] != 0:
                        votes[response] += 1/math.pow(neighbors[i][1], 2)
                    else:
                        votes[response] += 0
                else:
                    votes[response] += 1
            else:
                if self.weighted:
                    if neighbors[i][1] != 0:
                        votes[response] = 1/math.pow(neighbors[i][1], 2)
                    else:
                        votes[response] = 0
                else:
                    votes[response] = 1
        return sorted(votes.keys())[:1]


    def getNeighbors(self, newPoint):
        distances = []
        for i in range(len(self.trainingSet)):
            distances.append((self.trainingSet[i] , euc.getDistance(self.trainingSet[i], newPoint)))
        sortedDistances = sorted(distances, key=lambda tup: tup[1])
        return sortedDistances[:self.k]

    def getNewElementClass(self, newPoint):
        neighbors = self.getNeighbors(newPoint)
        return self.getClass(neighbors)


    def showClassErrors(self, classErrors, classNumbers):
        for i in classErrors.keys():
            print("Acerto na clase: " + str(i) + " " + str(classErrors[i]/classNumbers[i]))

    @profile
    def test(self, avaliationSet):
        bar = Bar('Processing', max=len(avaliationSet))
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

