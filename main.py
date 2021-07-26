import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
data = np.array(df)
n = df.shape[1]
records = df.shape[0]

alpha = 0.02
coefs = []
for i in range(0, n):
    print("Enter coef: ")
    coefs.append(float(input()))


y = []
for i in range(0, records):
    y.append(data[i][n-1])

x = np.zeros((records, n-1))
for i in range(0, n-1):
    for j in range(0, records):
        x[j][i] = data[j][i]


def h(a, b, x):

     HP = []
     for i in range(0, records):
         HP.append(a + (b * x[i]))

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


for i in range(700):
     a = theta0(a)
     b = theta1(b)


print(cost(a, b, x))
print(a)
print(b)

# plt.scatter(x, y)
# plt.show()