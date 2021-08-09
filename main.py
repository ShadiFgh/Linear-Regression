import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
n = df.shape[1]
m = df.shape[0]
# normalized = (df - df.mean()) / df.std()
learning_rate = 0.01
coefs = np.ones((1, n))
# print(coefs)
y = df.iloc[:, -1].values.reshape((m, 1))

# print(y)
# print(y.shape)

x1 = df.iloc[: , :-1]
x0 = np.ones((m, 1))
x = np.hstack((x0, x1))
# print(x)


def h(coefs, x, y):

     HP = np.matmul(x, coefs.T)

     return HP

# print(h(coefs, x, y))

def cost(coefs, x, y):

    diff = np.square(h(coefs, x, y) - y)
    total = sum(diff)
    return total/(2 * m)

# print(cost(coefs, x, y))


def theta(coefs):

    diff = h(coefs, x, y) - y
    mul = diff * x
    total = sum(mul)
    coefs -= (learning_rate * (total / m))

    return coefs
# print(theta(coefs))
c = []
for i in range(20000):
    coefs = theta(coefs)
    co = cost(coefs, x, y)[0]
    print(f"Lv: {i + 1} ==> Cost: {co}")
    c.append(co)


print("b:", coefs[0][0])
print("W:", coefs[0][1:])

plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.plot(c)
plt.show()