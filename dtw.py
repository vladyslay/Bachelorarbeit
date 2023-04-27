'''

# custom dtw algorithm
from dtw import *
import scipy
from scipy.spatial.distance import euclidean
import numpy as np

def dtw_table(x, y, distance=None):
    if distance is None:
        distance = euclidean
    nx = len(x)
    ny = len(y)
    table = np.zeros((nx+1, ny+1))
    
    #compute left column separately, j=0
    table[1:, 0] = np.inf
    
    #compute top row separately, i=0
    table[0, 1:] = np.inf
    
    for i in range(1, nx+1):
        for j in range(1, ny+1):
            d = distance(x[i-1], y[j-1])
            table[i, j] = d + min(table[i-1, j], table[i, j-1], table[i-1,j-1])
    return table

def dtw(x, y, table):
    i = len(x)
    j = len(y)
    path = [(i,j)]
    while i > 0 or j > 0:
        minval = np.inf
        if table[i-1][j-1] < minval:
            minval = table[i-1][j-1]
            step = (i-1, j-1)
        if table[i-1][j] < minval:
            minval = table[i-1][j]
            step = (i-1, j)
        if table[i][j-1] < minval:
            minval = table[i][j-1]
            step = (i, j-1)
        path.insert(0, step)
        i,j = step
    return np.array(path)

'''
#**************************************************************************************

import fastdtw
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np

x = np.array([1, 2, 3, 3, 7])
y = np.array([1, 2, 2, 2, 2, 2, 2, 4])

distance, path = fastdtw(x, y, dist=euclidean)

print(distance)
print(path)