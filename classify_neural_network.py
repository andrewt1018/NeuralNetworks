#Finds the separating line of 2 data sets
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
x1 = [-3.4, 0.5, 2.9, -0.1, -4.0, -1.3, -0.5, -4.1, -5.1, 1.9]
y1 = [6.2, 8.7, 2.1, 5.2, 2.2, 3.7, 7.5, 3.4, 1.6, 5.1]

x2 = [-2.0, -8.9, -4.2, -8.5, -6.7, -3.0, -5.3, -8.7, -7.1, -8.0]
y2 = [-8.4, 0.2, -7.7, -3.2, -4.0, -2.9, -6.7, -6.4, -9.7, -6.3]

x3 = [5.0, 11.2, 2.7, 4.2, 8.0, 8.8, 10.7, 6.4, 7.5, 3.2]
y3 = [0.4, -3.8, -9.1, -13.2, 0.6, -8.4, -12.1, -9.8, -1.9, -1.8]

x_total = [-3.0, 0.5, 2.9, -0.1, -4.0, -1.3, -0.5, -4.1, -5.1, 1.9, -2.0, -8.9, -4.2, -8.5, -6.7, -3.4, -5.3, -8.7, -7.1, -8.0, 5.0, 11.2, 2.7, 4.2, 1.0, 8.8, 10.7, 6.4, 7.5, 3.2]
y_total = [-2.9, 8.7, 2.1, 5.2, 2.2, 3.7, 7.5, 3.4, 1.6, 5.1, -8.4, 0.2, -7.7, -3.2, -4.0, 6.2, -6.7, -6.4, -9.7, -6.3, 0.4, -3.8, -9.1, -13.2, 1.6, -8.4, -12.1, -9.8, -1.9, 1.8]


x = []
y = []
for i in range(len(x_total)):
    x.append([x_total[i],y_total[i]])
print(x)
for i in range(10):
    y.append(0)
for i in range(10):
    y.append(1)
for i in range(10):
    y.append(2)

print(y)
clf = MLPClassifier(activation='relu', solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,2,), random_state=1)
clf.fit(x, y)
print(clf.predict([[2., 2.], [-6., -7.5], [7.5, -5]]))
print(clf.score(x,y))
print([coef.shape for coef in clf.coefs_])
print([coef for coef in clf.coefs_])

plt.scatter(x1, y1, color = 'r')
plt.scatter(x2, y2, color = 'b')
plt.scatter(x3, y3, color = 'g')
#plt.plot(x, y)
plt.show()