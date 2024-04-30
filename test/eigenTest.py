import numpy as np
import scipy as sp

arr = np.array([[1,0,0],[0,1,0],[0,0,1]])

res = np.linalg.eig(arr)

print(res)