import math
import numpy as np
import random


def sigmoid(val):
    return 1/(1 + pow(math.e, -val))

def relu(val):
    return max(0, val)

class NeuralNetwork:

    def forward_propagation(self, X):
        activations = X

        for l in range(1, len(self.layers)):
            activations = np.multiply(self.weights[l], activations) + self.biases[l]

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

            self.biases[l] = np.array([np.random.rand(num_neurons)]) # only one bias per neuron
            self.weights[l] = np.array([np.random.rand(num_neurons, num_prev_neurons)])
