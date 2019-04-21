import random


class Selection:

    def __init__(self, dataset, classColumn, nPrototypes=5):
        self.nPrototypes = nPrototypes
        self.dataset = dataset
        self.groups = [i[-1] for i in dataset]


    # Initial Selection Algorithms

    # Selects random elements
    def randomNSelectionEachClass(self, seed=None, inplace=False):
        random.seed(seed)
        elements = []
        classes = list(set(self.groups))
        for i in classes:
            for j in range(self.nPrototypes):
                randomElement = self.getRandomElementFromClass(i)
                elements.append(randomElement)
                if inplace:
                    self.dataset.remove(randomElement)
        return elements

    # Creates new elements by choosing random atributes from each of existing elements
    def newGenselection(self):
        nAtributos = len(self.dataset[0])
        elements = []
        for i in range(self.nPrototypes):
            element = []
            for j in range(nAtributos):
                element.append(self.selectRandomAtribute(self.dataset, j))
            elements.append(element)
        return elements



    # Utilities

    def getClass(self, point):
        return point[-1:][0]
        
    def getRandomElement(self):
        choice = random.choice(self.dataset)
        return choice

    def getRandomElementFromClass(self, requestedClass):
        choice = random.choice(self.dataset)
        while self.getClass(choice) != requestedClass:
            choice = random.choice(self.dataset)
        return choice
          
    def selectRandomAtribute(self,dataset, i):
        choice = random.choice(dataset)
        return choice[i]
        