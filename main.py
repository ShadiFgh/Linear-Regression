import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
n = df.shape[1]
m = df.shape[0]

alpha = 0.01
coefs = np.ndarray(n)

y = df.iloc[:, -1]
# print(y)

x = df.iloc[: , :-1]
x0 = np.ones((m, 1))
x = np.hstack((x0, x))
# print(x)


def h(coefs, x, y):

     HP = np.matmul(coefs, x.T)

     return HP
# print(h(coefs, x))

def cost(coefs, x, y):

    
    diff = np.square(h(coefs, x, y) - y)
    total = sum(diff)
    return total/(2*m)

# print(cost(coefs, x))



def theta(coefs):

    diff = h(coefs, x, y) - y
    for j in range(0, n):

        mul = diff * x[:, j]
        total = sum(mul)
        coefs[j] = coefs[j] - (alpha*(total/m))

    return coefs
# print(theta(coefs))

for i in range(1000):
     coefs = theta(coefs)


print(cost(coefs, x, y))
print(coefs)

