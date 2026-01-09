import numpy as np
from src.utils import sigmoid, sigmoid_derivative

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self._lr = lr
        
        # Initialize weights and biases randomly
        self._w1 = np.random.randn(hidden_size, input_size) * 0.01
        self._b1 = np.zeros((hidden_size, 1))
        
        self._w2 = np.random.randn(hidden_size, hidden_size) * 0.01
        self._b2 = np.zeros((hidden_size,1))
        
        self._w3 = np.random.randn(output_size, hidden_size) * 0.01
        self._b3 = np.zeros((output_size,1))

    def forward(self, x):
        """Go up layers -> return the predicted vector for
        the digit of that image"""
        #Matrix transformation
        #Hidden layer 1: W1.pixels
        self._z1 = np.dot(self._w1, x) + self._b1
        self._a1 = sigmoid(self._z1)
        
        #Hidden layer 2: W2.neurons_1
        self._z2 = np.dot(self._w2, self._a1) + self._b2
        self._a2 = sigmoid(self._z2)

        #Output layer: W3.neurons_2
        self._z3 = np.dot(self._w3, self._a2) + self._b3
        self._output = sigmoid(self._z3)

        return self._output #return the vector
    
    def backward(self, x, y_true):
        # propagated errors back the layers 
        #error of hidden 2 (layer 4)
        delta3 = (self._output - y_true) * sigmoid_derivative(self._output)

        # Hidden layer 1 error (layer 3)
        delta2 = np.dot(self._w3.T, delta3) * sigmoid_derivative(self._a2)

        # Input layer error (layer 2)
        delta1 = np.dot(self._w2.T, delta2) * sigmoid_derivative(self._a1)

        # Update weights and biases
        self._w3 -= self._lr * np.dot(delta3, self._a2.T) #learning rate to make the changes (jump) small
        self._b3 -= self._lr * delta3

        self._w2 -= self._lr * np.dot(delta2, self._a1.T)
        self._b2 -= self._lr * delta2

        self._w1 -= self._lr * np.dot(delta1, x.T)
        self._b1 -= self._lr * delta1

        