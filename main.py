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


def h(coefs, x):

     HP = np.zeros((records, 1))
     for j in range(0, records):
         for i in range(0, n - 1):
             HP[j] = HP[j] + coefs[i + 1] * x[j][i]
         HP[j] = HP[j] + coefs[0]

     return HP
# print(h(coefs, x))

def cost(coefs, x):

    diff = np.zeros((records, 1))
    for i in range(0, records):
        diff[i] = h(coefs, x)[i] - y[i]
    for i in range(0, records):
        diff[i] = diff[i] ** 2
    total = sum(diff)
    return total/(2*records)

# print(cost(coefs, x))

def theta0(a):
    diff = np.zeros((records, 1))
    for i in range(0, records):
        diff[i] = h(coefs, x)[i] - y[i]
    total = sum(diff)
    return a - (alpha*(total/records))

def theta(coefs):

    for j in range(1, n):
        diff = np.zeros((records, 1))
        for i in range(0, records):
            diff[i] = h(coefs, x)[i] - y[i]
        for i in range(0, records):
            diff[i] = diff[i] * x[i]
        total = sum(diff)
        coefs[j] = coefs[j] - (alpha*(total/records))
    return coefs


for i in range(700):
     coefs[0] = theta0(coefs[0])
     coefs = theta(coefs)


print(cost(coefs, x))
print(coefs[0])
print(coefs)

# plt.scatter(x, y)
# plt.show()