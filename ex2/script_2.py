# This script will train a Logistic Regression Model
# First let us have all the import statements
import screen
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from numpy import where
# import regression

# As always, first step is loading up the data
filepath = "/Users/ananthrajsingh/Desktop/Andrew_Ng_Python/ex2/ex2data1.csv"
# This is mndatory, else pandas will use first example as heading
names = ["marks1", "marks2", "admission"]
dataset = pd.read_csv(open(filepath), names = names, decimal = ",")
print(dataset.shape)
print(list(dataset.columns))
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
# Display spome of data
print("First 5 examples: ")
print(X[0:5, :])
print(y[0:5, :])


# Now we have arrays we can work on
# Let us visualize the data
pos, temp = where(y == 1)
print(pos)
neg, temp = where(y == 0)
print(neg)
# np.squeeze(np.asarray(M)) is used to convert matrix to 1D array,
# as scatter accepts 1d array.
print("shape of X[pos, 0]: ", np.squeeze(X[pos, 0]).shape)
plt.scatter(np.squeeze(np.asarray(X[pos, 0])), np.squeeze(np.asarray(X[pos, 1])), marker = "o", c = "r")
plt.scatter(np.squeeze(np.asarray(X[neg, 0])), np.squeeze(np.asarray(X[neg, 1])),  marker = "x", c = "b")
plt.show()

