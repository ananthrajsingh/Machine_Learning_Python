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
dataset = pandas.read_csv(open(filepath), decimal = ",")
dataset_array = dataset.values

# First two columns are features
X = dataset_array[:, 0:1]
# Third column contains values of y
y = dataset_array[:, 2]
print("Dataset loaded.")



