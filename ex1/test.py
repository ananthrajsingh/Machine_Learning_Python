import numpy as np
X = [[1,2,3],[1, 3, 1],[5,6,7]]
X = np.matrix(X)
print(X)
mean = X.mean(axis = 0)
print("Mean \n", mean)
diff = X - mean
print("X - mean \n", diff)