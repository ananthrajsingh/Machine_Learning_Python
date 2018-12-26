from subprocess import call
import clear_screen
import pandas
import matplotlib.pyplot as plt
import numpy as np
import cost


clear_screen.clear()
filepath1 = "/Users/ananthrajsingh/Desktop/Andrew_Ng_Python/ex1/ex1data1.csv"
names1 = ["profit", "population"]
filepath2 = "Users/ananthrajsingh/Desktop/Andrew_Ng_Python/ex1/ex1data2.csv"
dataset1 = pandas.read_csv(open(filepath1), names = names1, decimal = ",")

input("Press <ENTER> to continue")
clear_screen.clear()
print("Plotting data...")
array1 = dataset1.values
X = array1[:,0]
X = X.astype(np.float) # Since array was made of String
y = array1[:,1]
y = y.astype(np.float)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(X,y, label = "Data")
plt.draw()
fig.show()


#####################################################################
# Cost and Gradient Descent
#####################################################################
clear_screen.clear()
input("Press <ENTER> to calculate Cost and Gradient Descent")
# getting dimensions of X
X = np.matrix(X)
X = X.transpose()
y = np.matrix(y)
y = y.transpose()
n,m = X.shape
# Make an column of 1s of size m
zeros = np.ones((n,1))
# Adding a column of 1's to X to add bias unit
X = np.hstack((zeros, X))

# there will be 2 theta parameters
theta = np.zeros((2, 1), dtype = "float")
iterations = 1500
alpha = 0.01


print("Testing cost function")
J = cost.computeCost(X, y, theta)
print('With theta = [0 ; 0]\nCost computed = ', J[0][0])
print('Expected cost value (approx) 32.07\n')

J = cost.computeCost(X, y, [[-1], [2]])
print('\nWith theta = [-1 ; 2]\nCost computed = ', J[0][0])
print('Expected cost value (approx) 54.24\n');

input("Program paused. Press <ENTER> to continue")

print("Running Gradient Descent...")
num_iters = 1500
J_history, theta = cost.gradientDescent(X, y, theta, alpha, num_iters)
print("Theta found by gradientDescent")
print(theta)
print("Expected theta values (approx)")
print(" -3.6303 \n 1.1664")
# X is m x 2 and theta is 2 x 1
# theta_t = np.transpose(theta)
ax1.scatter([X[:,1]], [X * theta],color = "k", marker = ".", label = "Fit")
plt.draw()
fig.show()
plt.legend(loc='best')
# plt.show()
input("Press <ENTER> to continue")

#####################################################################
# Visualizing Data
#####################################################################
# theta0_values = np.arange(-10, 10, 0.2)
# theta1_values = np.arange(-1, 4, 0.05)
# J_vals = np.zeros((len(theta0_values), len(theta1_values)), dtype = "float")


# for i in range(len(theta0_values)):
# 	for j in range(len(theta1_values)):
# 		t = [[theta0_values[i]],[theta1_values[j]]
# 		J_vals[i][j] = cost.computeCost(X, y, t);
# print(J_vals)


