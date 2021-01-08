import numpy as np

instances = 10000

sequences = 10

X = np.zeros((instances, sequences * 2))

X[:int(instances / 2), 1] = 1

for i in range(instances):

    for j in range(0, sequences * 2, 2):

        X[i, j] = np.random.random()

Y = np.zeros((instances, 1))

for i in range(len(Y)):

    if X[i, 1] == 0:

        Y[i] = X[i, -2] * 3

    if X[i, 1] == 1:

        Y[i] = X[i, -2] * 9


X_lstm = X.reshape(X.shape[0], X.shape[1] // 2, 2)

print("X = ", X[0][1])
print("X = ", X[0])
# print("Y = ", Y)