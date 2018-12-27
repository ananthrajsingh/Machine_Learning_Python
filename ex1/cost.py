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

# This returns normalised values for each feature with mean 0 and
# standard deviation of 1. This is a good preprocessing practise.
# Here we will calculate mean for each feature, and subtract from 
# each example of that feature, then storing value of mean in mu.
# Next we eill calculate standard deviation for each feature. We 
# will divide that feature values with this deviation, storing it 
# in sigma.
def featureNormalize(X):
	mu = X.mean(axis = 0)
	X_norm = X - mu
	# axis = 0 means we need to calculate along columns
	# ddof = 1 means we will divide with N - 1 rather than N
	# actually (N - ddof), we are doing this to match the result with
	# Matlab result. Python takes ddof = 0 as default.
	sigma = X_norm.std(axis = 0, ddof = 1) # 0 means along columns
	X_norm = X_norm/sigma
	return (X_norm, mu, sigma)

