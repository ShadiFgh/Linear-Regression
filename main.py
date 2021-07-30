import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
n = df.shape[1]
records = df.shape[0]

alpha = 0.02
coefs = np.ndarray(n)

y = df.iloc[:, -1]
# print(y)

x = df.iloc[: , :-1]
x0 = np.ones((records, 1))
x = np.hstack((x0, x))
# print(x)


def h(coefs, x):

     HP = np.matmul(coefs, x.T)

     return HP
# print(h(coefs, x))

def cost(coefs, x):

    diff = (h(coefs, x) - y) * (h(coefs, x) - y)
    total = sum(diff)

    return total/(2*records)

# print(cost(coefs, x))



def theta(coefs):

    diff = h(coefs, x) - y
    for j in range(0, n):

        mul = diff * x[:, j]
        total = sum(mul)
        coefs[j] = coefs[j] - (alpha*(total/records))

    return coefs
# print(theta(coefs))

for i in range(700):
     coefs = theta(coefs)


print(cost(coefs, x))
print(coefs)

