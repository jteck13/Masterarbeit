import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def relu(x):
    return np.maximum(0,x)

def leaky(x):
    alpha = 0.1
    return np.maximum(alpha * x, x)

def softmax_function(x):
    z = np.exp(x)
    z_ = z/z.sum()
    return z_


x = np.linspace(-10, 10)


plt.plot(x, sigmoid(x))
plt.axis('tight')
plt.title('Sigmoid activation function')
plt.show()

x = np.linspace(-10, 10)
plt.plot([0.8, 1.2, 3.1], softmax_function([0.8, 1.2, 3.1]))
plt.axis('tight')
plt.title('Softmax activation function')
plt.show()

x = np.linspace(-10, 10)
plt.plot(x, relu(x))
plt.axis('tight')
plt.title('ReLu activation function')
plt.show()

x = np.linspace(-10, 10)
plt.plot(x, leaky(x))
plt.axis('tight')
plt.title('Leaky-ReLu activation function')
plt.show()








