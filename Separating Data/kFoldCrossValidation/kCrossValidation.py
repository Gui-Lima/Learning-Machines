import random
from os import path

class kFoldCrossValidation:

    def __init__(self, dataset, k, validationColumn, debug=True, seed = None):
        self.dataset = dataset
        self.k = k
        self.debug = debug
        self.validationColumn = validationColumn
        self.groups = dataset[validationColumn].tolist()
        self.seed = seed

    def writeFoldsInFile(self, folds):
        f = open(path.abspath("Separating Data/kFoldCrossValidation/Folds.txt"), "w")
        f.write("Fold Sizes: " + str(len(folds[0])))
        f.write('\n')
        f.write('\n')
        for i in range(len(folds)):
            f.write("------------------------------------")
            f.write("Fold " + str(i+1) + " : ")
            f.write('\n')
            for j in range(len(folds[i])):
                f.write(str(folds[i][j]))
                f.write('\n')
            f.write('\n')
        f.close()

    def writeSetsInFile(self, trainingSet, AvaliationSet):
        f = open(path.abspath("Separating Data/kFoldCrossValidation/TrainingAndAvaliationSets.txt"), "w")
        f.write("Training sets size : " + str(len(trainingSet[0])))
        f.write('\n')
        f.write("Avaliation sets size : " + str(len(AvaliationSet[0])))
        f.write('\n')
        f.write('\n')
        for i in range(len(trainingSet)):
            f.write("In the " + str(i) + " iteration of the kfold, the training set will be : ")
            f.write('\n')
            for j in range(len(trainingSet[i])):
                f.write(str(trainingSet[i][j]) + " ,")

            f.write('\n')
            f.write(" And the avaliation set will be: ")
            f.write('\n')

            for j in range(len(AvaliationSet[i])):
                f.write(str(AvaliationSet[i][j]) + " ,")

            f.write('\n')
            f.write('\n')
        f.close()


    def getPercentagesOfEachElement(self, l):
        percentages = {}
        elements = list(set(l))
        for i in range(len(elements)):
            element = elements[i]
            percentage = l.count(element) / len(l)
            percentages[element] = percentage
            print("This group: " + str(element) + " represents " + str(percentage) + " of the database")
        return percentages

    def run(self, traningSets=[], avaliationSets=[], stratified=True):
        print("-------------------------------")
        print("Making Cross Validation")
        datasetSize = len(self.groups)
        setSize  = round(datasetSize/self.k)
        print("The dataset size is " + str(datasetSize))
        print("Since its " + str(self.k) + " folds, each fold will have " + str(setSize) + " elements, selected stratified = " + str(stratified))
        if stratified:
            folds = self.makeRandomFolds()
        else:
            folds = self.makeFolds()

        for i in range(self.k):
            trainingfold = []
            selectedOutFold = i
            for j in range(len(folds)):
                if j is not selectedOutFold:
                    trainingfold.extend(folds[j])
            traningSets.append(trainingfold)
            avaliationSets.append(folds[selectedOutFold])

        if self.debug:
            self.writeSetsInFile(traningSets, avaliationSets)
        print("-------------------------------")


    def makeRandomFolds(self):
        folds = []
        setSize  = round(len(self.groups)/self.k)
        percentagesOfEachGroup = self.getPercentagesOfEachElement(self.groups)
        groupAndIndex=[]
        for i in range(len(self.groups)):
            groupAndIndex.append((self.groups[i], i))
        eachGroup = []
        groupSets = list(set(self.groups))
        ngroups = len(groupSets)
        groupAndNumberOfElementsInEachFold = {}
        for i in range(ngroups):
            group = groupSets[i]
            eachGroup.append([x for x in groupAndIndex if x[0] == group])
            percentageOfThisGroupInTheDataset = percentagesOfEachGroup[group]
            numberOfElementsOfThisGroup = setSize * percentageOfThisGroupInTheDataset
            groupAndNumberOfElementsInEachFold[group] = round(numberOfElementsOfThisGroup)
            print("Since each fold size is " + str(setSize) + " and the group " + str(group) +  " representation in the dataset is " + str(percentageOfThisGroupInTheDataset) + ", this group will have " + str(numberOfElementsOfThisGroup) + " elements")    
        
        for i in range(self.k):
            fold = []
            print("Making fold " + str(i + 1))
            for j in range(ngroups):
                print("Elements of this fold on group " + str(groupSets[j]) + " : " + str(groupAndNumberOfElementsInEachFold[groupSets[j]]))
                fold.extend(self.getRandomElements(eachGroup[j], groupAndNumberOfElementsInEachFold[groupSets[j]]))
            folds.append(fold)

        for i in range(len(eachGroup)):
            if(len(eachGroup[i]) > 0):
                for j in range(len(eachGroup[i])):
                    folds[len(folds) - 1].extend(eachGroup[i][j])

        if self.debug:
            self.writeFoldsInFile(folds)
        return folds

    def getRandomElements(self, l, n):
        random.seed(self.seed)
        elements  = []
        for i in range(n):
            if(len(l) != 0):
                choice = random.choice(l)
                l.remove(choice)
                elements.append(choice)
        return elements

    def makeFolds(self):
        # TODO
        return []
