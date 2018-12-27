# Linear Regression with multiple variables
# We have a house to sell.
# How do we know how much we will fetch for it?
# Fortunately we have data for houses, we can train a linear 
# regression model.
# Let us first import important libraries
import numpy as np 
import clear_screen
import matplotlib.pyplot as plt 
import cost
import pandas
from subprocess import call

clear_screen.clear()
# First thing first, load the data.
filepath = "/Users/ananthrajsingh/Desktop/Andrew_Ng_Python/ex1/ex1data2.csv"
# NOTE: We need to add these names!
# Else python will take first example set (1st row) as heading
# Decreasing number of examples by 1
names = ["area", "bedroom", "price"]
dataset = pandas.read_csv(open(filepath), names = names, decimal = ",")
dataset_array = dataset.values
print("Dataset shape ", dataset_array.shape)
dataset_array = np.matrix(dataset_array)

# First two columns are features
X = dataset_array[:, 0:2]
print("X shape ", X.shape)
# Third column contains values of y
y = dataset_array[:, 2]
print("y shape ", y.shape)
# These arrays have strings, let's convert them to float
X = X.astype(np.float)
y = y.astype(np.float)

# m has number of examples that we have, i.e. number of columns
m, n = X.shape

# Better to convert arrays to numpy matrices
X = np.matrix(X)
y = np.matrix(y)

print("Dataset loaded in X and y.")

# Let us show user some data
print("First 10 examples from dataset: ")
print(X[0:9, :])
print(y[0:9, :])

input("Program paused. Press <enter> to continue.")

#####################################################################
# FEATURE NORMALIZATION
#####################################################################

print("Normalizing features...")

X, mu, sigma = cost.featureNormalize(X)
# print("mean mu: ", mu)
# print("standard deviation sigma: ", sigma)
# print("First 10 examples of Normalised X:")
# print(X[0:9, :])

# input("Program paused. Press <enter> to continue.")

#####################################################################
# GRADIENT DESCENT
#####################################################################

# Add intercept term to our X
# m is the number of examples we have
# creating vector of ones
intercept_ones = np.ones((m,1))
# Stick this vector to our X
X = np.hstack((intercept_ones, X))
# shape of X is now m x n+1
print("Running Gradient Descent...")
# Let us choose an alpha and number of iterations
alpha = 0.01
iters = 400
# Also initialise theta to pass as argument
theta = np.zeros((n + 1,1), dtype = "float")
# We will be using same gradient descent function
# since we implemented it in vectorised form,
# it will work for multiple features too.
J_history, theta = cost.gradientDescent(X, y, theta, alpha, iters)
print("Theta calculated from gradientDescent: ", theta)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(J_history)
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
print("Close plot figure to continue")
plt.show()

print("Predicted price of a 1650 sq-ft, 3 bedrooms house: ")
# print("mu :", mu)
# print("sigma :", sigma)
normalized_area = (1650 - mu[0,0])/sigma[0,0]
normalised_br = (3 - mu[0,1])/sigma[0,1]
price = theta[0] + normalized_area*theta[1] + normalised_br*theta[2]
print(price)




