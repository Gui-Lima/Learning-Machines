import math

def index_data(data):
    for i in range(len(data)):
        data[i] = (i, data[i])
    return data

def kdtree(data):
    quant_att = len(data[0])
    data = index_data(data)
    return kdtree_recursion(data, quant_att, 0)

def mean_column(data, itera):
    total = 0
    for i in data:
        total += i[1][itera]
    return float(total)/len(data)

def kdtree_recursion(data, quant_att, itera):
    left = []
    right = []
    mid = mean_column(data, itera)
    if len(data) == 1:
        return {"end": data[0]}
    for i in data:
        if i[1][itera]<= mid:
            left.append(i)
        else:
            right.append(i)
    if itera == quant_att-1:
        return {mid: {"left": {"end": left} , "right": {"end": right}} }
    elif len(left)!=0 and len(right)!=0:
        tree = {}
        tree[mid] = {"left": kdtree_recursion(left, quant_att, itera+1) , "right": kdtree_recursion(right, quant_att, itera+1)}
        return tree

def find_neighbors(model, instance, itera, k):
    mid = list(model.keys())[0]
    if instance[itera] <= mid:
        model = model[mid]["left"]
    elif instance[itera] > mid:
        model = model[mid]["right"]
    if list(model.keys())[0] == "end":
        return knn(model["end"], instance, k)
    else:
        return find_neighbors(model, instance, itera+1, k)

def euclidean_dist(p1, p2):
    if len(p1) != len(p2):
        raise Exception("Length of two numbers in euclidean distance must be equal")
    else:
        aux = []
        for i in range(len(p1)):
            temp = (p1[i]-p2[i])**2
            aux.append(temp)
        total = sum(aux)
        return math.sqrt(total)

def knn(neighbors, instance, k):
    calc_dist = lambda x: euclidean_dist(instance, x[1])
    neighbors.sort(key=calc_dist)
    return neighbors[:k]
    

data = [[0.3, 0.7, 3, 5, 0.1], [0.3, 0.7, 3, 5, 0.1], [0.5, 0.8, 6, 5, 0.3],[0.8, 0.5, 7, 4, 0.5],[0.3, 0.8, 23, 7, 0.2],[0.1, 1.3, 9, 2, 1],
        [13, 5, 101, 0.3, 3],   [8, 2, 51, 0.8, 2],    [6, 3, 30, 1.5, 5],    [3, 1.5, 130, 0.1, 4],  [30, 2.3, 27, 1, 2]]
a = kdtree(data)
print(find_neighbors(a, [0.3, 0.7, 3, 5, 0.1], 0, 3))

