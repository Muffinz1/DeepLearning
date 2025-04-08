# First Lab Deeplearning
# imports
import numpy as np


# Defining w and b
weights = np.around(np.random.uniform(size=6), decimals=2)
biases = np.around(np.random.uniform(size=3), decimals=2)

# Printing w and b
print(weights)
print(biases)

x_1 = 0.5 # input 1
x_2 = 0.85 # input 2


# printing the x
print('x1 is {} and x2 is {}'.format(x_1, x_2))


# computing the weighted sum of the inputs "x1*w1+x2*w2+b = z"
z_11 = x_1 * weights[0] + x_2 * weights[1] + biases[0]

# Computing the weighted sum of the inputs for the second nodes
z_12 = x_1 * weights[2] + x_2 * weights[3] + biases[1]

print('The weighted sum of the inputs at the second node in the hidden layer is {}'.format(np.around(z_12, decimals=4)))

# activation "Sigmoid function" For first node

a_11 = 1.0 / (1.0 + np.exp(-z_11))
print('The activation of the first node in the hidden layer is {}'.format(np.around(a_11, decimals=4)))

# activation "Sigmoid function" For second node
a_12 = 1.0 / (1.0 + np.exp(-z_12))

print('The activation of the second node in the hidden layer is {}'.format(np.around(a_12, decimals=4)))


z_2 = a_11 * weights[4] + a_12 * weights[5] + biases[2]

# The output of the network for x1 = 0.5 and x2 = 0.85 is
a_2 = 1.0 / (1.0 + np.exp(-z_2))
print('The output of the network for x1 = 0.5 and x2 = 0.85 is {}'.format(np.around(a_2, decimals=4)))

