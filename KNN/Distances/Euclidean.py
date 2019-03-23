import math

def getDistance(ex1, ex2):
    distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(ex1, ex2)]))
    return distance