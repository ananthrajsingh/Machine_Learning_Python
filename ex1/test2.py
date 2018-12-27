import numpy as np
X = [[2,3,4],[8,7,1]]
X = np.matrix(X)
print("X : \n", X)
std = [2,3,1]
print("std : \n", std)
print("X/std : \n")
Y = X/std
print(Y)