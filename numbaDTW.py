import numpy as np
import numba
import time as ttt
from numba import cuda, float64


# ############################################################################
# #######################regular accelerated dtw##############################
# ############################################################################
#
#
def distanceMatrix(x, y):
    x = np.array(x) # pitches in np array
    y = np.array(y)
  
    #to mesure the time taken 
    oldTime = ttt.time()
    #matrix not array, transpose of x 
    distance = __distanceMatrix(x[:, np.newaxis], y)
    #print('distance time ' + str(ttt.time() - oldTime))
    return distance

#works in parallel using numba
@numba.vectorize(['float64(float64, float64)'], target='parallel')
def __distanceMatrix(x, y):
    #musical meaning
   # return (abs(x - y))
		return abs(x - y)#, abs(2*x - y), abs(x - 2*y), abs(x - 4*y), abs(4*x - y), abs(y - 8*x), abs(x - 8*y))
