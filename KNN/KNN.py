import sys
from os import path
sys.path.insert(0, path.abspath('KNN/Distances/'))
import Euclidean as euc
from progress.bar import Bar

class Knn:

    def __init__(self, trainingSet, k):
      self.trainingSet = trainingSet
      self.k = k

    def getClass(self, neighbors):
        votes = {}
        for i in range(len(neighbors)):
            response = neighbors[i][0][-1]
            if response in votes:
                votes[response] += 1
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

    def test(self, avaliationSet):
        bar = Bar('Processing', max=len(avaliationSet))
        corrects = 0
        for i in range(len(avaliationSet)):
            example = avaliationSet[i]
            actualClass = example[-1]
            #print("new example to be avaliated :" + str(example))
            #print("The actual class of this example is : " + str(actualClass))
            predictedClass = self.getNewElementClass(example)
            #print("Predicted class is : " + str(predictedClass[0]))
            if (actualClass == predictedClass[0]):
                corrects += 1
            bar.next()
        bar.finish()
        print("Of " + str(len(avaliationSet)) + " examples, we got " + str(corrects) + " of them right!")
        print("Final percentage correctness :" + str(corrects/len(avaliationSet)))
        return corrects/len(avaliationSet)
