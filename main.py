import sklearn
from sklearn import model_selection
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
import random

#y = mx +c
#F = 1.8*C +32

x = list(range(0, 100)) # C
y = [1.8 * F + 32 for F in x] # F
print(f'X: {x}')
print(f'Y: {y}')

plt.plot(x, y, '-*r')
plt.show()

x = np.array(x).reshape(-1,1)
y = np.array(y).reshape(-1,1)

xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
model = linear_model.LinearRegression()
model.fit(xTrain, yTrain)
print(f'Intercept: {model.intercept_}')
print(f'Coefficient: {model.coef_}')

acc = model.score(xTest, yTest)
print(f'Accuracy: {round(acc*100, 2)}%')


