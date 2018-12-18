from subprocess import call
import clear_screen
import pandas
import matplotlib.pyplot as plt
import numpy as np

# command = "Users/ananthrajsingh/Desktop/Andrew_Ng_Python/ex1/clear_screen.py"
# call(command, shell = True)
clear_screen.clear()
print("Pausing the script")
input("Press <ENTER> to continue")
print("Continuing...")
filepath1 = "/Users/ananthrajsingh/Desktop/Andrew_Ng_Python/ex1/ex1data1.csv"
names1 = ["profit", "population"]
filepath2 = "Users/ananthrajsingh/Desktop/Andrew_Ng_Python/ex1/ex1data2.csv"
dataset1 = pandas.read_csv(open(filepath1), names = names1, decimal = ",")
print("Printing dataset")
print(dataset1)
input("Press <ENTER> to continue")
clear_screen.clear()
print("Plotting data...")
array1 = dataset1.values
X = array1[:,0]
X = X.astype(np.float)
y = array1[:,1]
y = y.astype(np.float)
# print("Printing X")
# print(X)
# input("Press <ENTER> to continue")
# clear_screen.clear()
# print("Printing y")
# print(y)
# y = array1[:,1]
# fig = plt.figure()
plt.scatter(X,y)
# plt.axis([0, 5, 0, 5])
# fig.suptitle("Profit vs population")
plt.show()
