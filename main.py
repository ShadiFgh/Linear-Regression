import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
data = np.array(df)
n = df.shape[1] - 1
records = df.shape[0]

a = 0.01
b = 1
alpha = 0.02

y = []
for i in range(0, 96):
    y.append(data[i][1])

x = []
for i in range(0, 96):
    x.append(data[i][0])

def h(a, b, x):

     HP = []
     for i in range(0, 96):
         HP.append(a + b * x[i])

     return HP


def cost(a, b, x):

    diff = np.subtract(h(a, b, x), y)
    for i in range(0, 96):
        diff[i] = diff[i] ** 2
    total = sum(diff)
    return total/(2*records)


def theta0(a):

    diff = np.subtract(h(a, b, x), y)
    total = sum(diff)
    return a - (alpha*(total/records))

def theta1(b):
    diff = np.subtract(h(a, b, x), y)
    for i in range(0, 96):
        diff[i] = diff[i] * x[i]
    total = sum(diff)
    return b - (alpha*(total/records))


for i in range(1000):
     a = theta0(a)
     b = theta1(b)


print(cost(a, b, x))
print(a)
print(b)

plt.scatter(x, y)
plt.show()