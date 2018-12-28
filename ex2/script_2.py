# This script will train a Logistic Regression Model
# First let us have all the import statements
import screen
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from numpy import where
import regression
screen.clear()
# As always, first step is loading up the data
filepath = "/Users/ananthrajsingh/Desktop/Andrew_Ng_Python/ex2/ex2data1.csv"
# This is mndatory, else pandas will use first example as heading
names = ["marks1", "marks2", "admission"]
dataset = pd.read_csv(open(filepath), names = names, decimal = ",")
# print(dataset.shape)
# print(list(dataset.columns))
dataset_array = dataset.values
dataset_array = np.matrix(dataset_array)
# Let us separate X and y from this array
X = dataset_array[:, 0:2]
y = dataset_array[:, 2]

X = X.astype(np.float)
y = y.astype(np.float)
# Better to convert these to numpy matrices
X = np.matrix(X)
y = np.matrix(y)

m,n = X.shape
# Display spome of data
print("First 5 examples: ")
print(X[0:5, :])
print(y[0:5, :])

input("Program paused. Press <enter> to see data plot.")


#####################################################################
# VISUALIZE DATA
#####################################################################

# Now we have arrays we can work on
# Let us visualize the data
pos, temp = where(y == 1)
neg, temp = where(y == 0)
# np.squeeze(np.asarray(M)) is used to convert matrix to 1D array,
# as scatter accepts 1d array.
plt.scatter(np.squeeze(np.asarray(X[pos, 0])), np.squeeze(np.asarray(X[pos, 1])), marker = "o", c = "r")
plt.scatter(np.squeeze(np.asarray(X[neg, 0])), np.squeeze(np.asarray(X[neg, 1])),  marker = "x", c = "b")
print("Close plot window to continue.")
plt.show()


#####################################################################
# COMPUTE COST AND GRADIENT
#####################################################################

# Let us first add intercept term
intercept_ones = np.ones((m,1))
X = np.hstack((intercept_ones, X))

# Initialize theta
initial_theta = np.zeros((n+1,1))

# let us calculate the cost and gradient
cost, gradient = regression.costFunction(initial_theta, X, y)

print('Cost at initial theta (zeros): \n', cost);
print('Expected cost (approx): 0.693\n');
print('Gradient at initial theta (zeros):');
print(gradient);
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n');

test_theta = [[-24], [0.2], [0.2]];
test_theta = np.matrix(test_theta)
cost, grad = regression.costFunction(test_theta, X, y);

print('\nCost at test theta: \n', cost);
print('Expected cost (approx): 0.218\n');
print('Gradient at test theta:');
print('\n', grad);
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n');


