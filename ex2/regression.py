# This will have all the helper functions for this exercise
# Here goes the import statements
import numpy as np 
import math
def sigmoid(z):
	return 1/(1 + np.exp(-z))

# This will be used to calculate cost and gradient for logistic regression

def costFunction(theta, X, y):
	# First let us get number of examples we are dealing with
	m,n = y.shape
	a,b = theta.shape
	# Initialize cost and gradient
	# J = 0
	# grad = np.zeros((a,b))
	
	print("Shape of theta")
	print(theta.shape)
	print("Shape of X")
	print(X.shape)
	print("Shape of y")
	print(y.shape)
	dot1 = np.dot(X, theta)
	dot2 = np.dot(X, theta)
	A = np.dot(np.log(np.transpose(sigmoid(dot1))), (-y)) - np.dot(np.log(1 - np.transpose(sigmoid(dot2))), (1-y))
	J = A/m
	grad = (np.transpose(X) * (sigmoid(X * theta) - y))/m
	return (J, grad)

def onlyCost(theta, X, y):
	m,n = y.shape
	dot1 = np.dot(X, theta)
	dot2 = np.dot(X, theta)
	sigmoid1 = sigmoid(dot1)
	sigmoid2 = sigmoid(dot2)
	transpose1 = np.transpose(sigmoid1)
	transpose2 = np.transpose(sigmoid2)
	log1 = np.log(transpose1)
	log2 = np.log(1 - transpose2)
	A = np.dot(log1, (-y)) - np.dot(log2, (1-y))
	J = A/m
	return J
	# r = J[0]
	# if np.isnan(r):
	# 	return np.inf
	# return r
def onlyGradient(theta, X, y):
  	m,n = y.shape
  	# print("Shape of theta")
  	# print(theta.shape)
  	# print("Shape of X")
  	# print(X.shape)
  	# print("Shape of y")
  	# print(y.shape)
  	dot1 = np.dot(X, theta)
  	# print("Shape of dot1")
  	# print(dot1.shape)
  	sigmoid1 = sigmoid(dot1)
  	# print("Shape of sigmoid1")
  	# print(sigmoid1.shape)
  	transpose1 = np.transpose(X)
  	# print("Shape of transpose1")
  	# print(transpose1.shape)
  	# print("Shape of (sigmoid1 - y)")
  	# print((sigmoid1 - y).shape)
  	# print("Shape of y")
  	# print(y.shape)

  	grad = np.dot(transpose1, (sigmoid1 - y))/m
  	# print("grad from onlyGradient- ", grad)
  	return np.squeeze(np.asarray(grad))
  	# return grad
	