#This functions takes in X, y and theta and returns the cost of these parameters over X set
import numpy as np
def computeCost(X, y, theta):
	m,n = y.shape
	# print("Shape of y - ")
	# print(y.shape)
	# print(m)
	J = 0
	# Calculating cost in vectorised form
	# X = np.matrix(X)
	# print(X)
	# y = np.matrix(y)
	# print(y.shape)
	# print("In Cost FUnction")
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
	# print("sum - ")
	# print(sum1)
	return sum1[0]/(2*m)

# This functions performs gradient descent
# This will ideally reduce the cost of theta with each iteration. To learn theta.
# As we iterate, a point will come where values will start to converge insignificantly.
# That is where we should stop.
# But here we will iterate through a predefined number of iterations.
# This works on the principal, the lower the cost, lower is deviation from expected values.
def gradientDescent(X, y, theta, alpha, num_iters):
	m,n = y.shape # m has number of rows
	J_history = np.zeros((num_iters, 1), dtype = "float") # This will store all previous cost values as we iterate
	X_transpose = np.transpose(X)
	for i in range(num_iters):
		temp = X_transpose * ((X * theta) - y)
		delta = temp/m
		theta = theta - alpha * delta
		J_history[i] = computeCost(X, y, theta)
	return (J_history, theta)



