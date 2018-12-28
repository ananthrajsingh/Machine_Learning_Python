# This will have all the helper functions for this exercise
# Here goes the import statements
import numpy as np 
import math
def sigmoid(z):
	return 1/(1 + np.exp(z))

# This will be used to calculate cost and gradient for logistic regression

def costFunction(theta, X, y):
	# First let us get number of examples we are dealing with
	m,n = y.shape
	a,b = theta.shape
	# Initialize cost and gradient
	J = 0
	grad = np.zeros((a,b))
	A = np.log(np.transpose(sigmoid(X * theta))) * (-y) - np.log(1 - np.transpose(sigmoid(X * theta))) * (1-y)
	J = A/m
	grad = (np.transpose(X) * (sigmoid(X * theta) - y))/m
	return (J, grad)