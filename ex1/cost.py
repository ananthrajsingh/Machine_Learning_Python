#This functions takes in X, y and theta and returns the cost of these parameters over X set
import numpy as np
def computeCost(X, y, theta):
	m,n = y.shape
	print("Shape of y - ")
	print(y.shape)
	print(m)
	J = 0
	# Calculating cost in vectorised form
	# X = np.matrix(X)
	# print(X)
	# y = np.matrix(y)
	# print(y.shape)
	print("In Cost FUnction")
	# print("Shape of y")
	# print(y.shape)
	# print("Shape of X")
	# print(X.shape)
	# theta = theta.astype(np.float)
	# theta = np.matrix(theta)
	# theta = theta.transpose
	# print("THETA - ")
	# print(theta)
	# print("X - ")
	# print(X)

	A = np.square(X * theta - y)
	# print("A - ")
	# print(A)
	
	sum1 = A.sum(axis = 0)[0] #Columwise sum will return array, taking its first term
	print("sum - ")
	print(sum1)
	return sum1[0]/(2*m)

