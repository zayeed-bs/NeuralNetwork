import math
import numpy as np
import random


def sigmoid(z):
    return 1/(1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

class NeuralNetwork:
    def forward_propagation(self, X):
        activations = X.reshape(-1,1)

        for l in range(1, len(self.layers)):
            z = np.dot(self.weights[l], activations) + self.biases[l]

            if l == len(self.layers) - 1:  # Output layer
                activations = sigmoid(z)  # For classification
            else:  # Hidden layers
                activations = relu(z)

        return activations

    def __init__(self, layers:list, learning_rate:float):
        # Initialize Properties
        self.layers = layers
        self.learning_rate = learning_rate

        # Initialize the weights and biases
        self.weights = {}
        self.biases = {}

        for l in range(1, len(layers)):
            num_neurons = layers[l]
            num_prev_neurons = 0 if l == 0 else layers[l - 1]

            self.biases[l] = np.random.rand(num_neurons,1) # only one bias per neuron
            self.weights[l] = np.random.rand(num_neurons, num_prev_neurons) * 0.01
