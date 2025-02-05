import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1/(1 + np.exp(-z))

def der_sigmoid(z):
    return np.dot(sigmoid(z), (1-sigmoid(z)).T)

def relu(z):
    return np.maximum(0, z)

class NeuralNetwork:
    def forward_propagation(self, X):
        activations = {0: X.reshape(-1,1)}
        zs = {}

        for l in range(1, len(self.layers)):
            z = np.dot(self.weights[l], activations[l-1]) + self.biases[l]
            zs[l] = z

            if l == len(self.layers)-2:  # Output layer
                activations[l] = sigmoid(z)  # For classification
            else:  # Hidden layers
                activations[l] = relu(z)

        return activations, zs

    def calculate_loss(self, Y, Y_pred):
        return np.mean((Y-Y_pred)**2)

    def backward_propagation(self, X, Y, activations:dict, zs):
        Y = Y.reshape(-1,1)
        L = len(self.layers)-1
        m = len(X)
        grads = {"dW": {}, "dB": {}}

        for l in range(L, 0, -1):
            d_loss_d_aL = 2*(activations[l]-Y)
            d_aL_d_zL = der_sigmoid(zs[l])
            delta_L = (d_loss_d_aL * d_aL_d_zL.T).reshape(-1, m)

            dW_L = np.dot(delta_L, activations[l-1].T).reshape(-1,1) / m
            dB_L = np.sum(delta_L, axis=1, keepdims=True) / m

            grads["dW"][l] = dW_L
            grads["dB"][l] = dB_L

        return grads


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

            self.biases[l] = np.random.randn(num_neurons,1) # only one bias per neuron
            self.weights[l] = np.random.randn(num_neurons, num_prev_neurons) * 0.01 # multiply by 0.01
                                                                                   # to make sure the value
                                                                                   # isn't batshit crazy

