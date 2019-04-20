from enum import Enum

class knnTypes(Enum):
    NORMAL = 0
    WEIGHTED = 1
    ADAPTATIVE = 2


MIN_NUMBER = 0.00000001



# List Ops
class listOps:
    def listSum(self, l1, l2):
        return [sum(pair) for pair in zip(l1,l2)]

    def listMult(self, l1, l2):
        return [x * y for x, y in zip(l1, l2)]

    def listSub(self, l1, l2):
        return [x - y for x, y in zip(l1, l2)]

    def scalarMultList(self, scalar, l1):
        for i in range(len(l1)):
            l1[i] = l1[i] * scalar
        return l1