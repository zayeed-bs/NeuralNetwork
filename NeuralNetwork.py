import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1/(1 + np.exp(-z))

def der_sigmoid(z):
    return np.dot(sigmoid(z), (1-sigmoid(z)).T)

def relu(z):
    return np.maximum(0, z)

def der_relu(z):
    return np.where(z>0, 1, 0)

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

        d_loss_d_aL = 2 * (activations[L] - Y)

        for l in range(L, 0, -1):
            d_aL_d_zL = der_sigmoid(zs[l]) if l == L else der_relu(zs[l])
            delta_L = d_loss_d_aL * d_aL_d_zL

            if l > 1:  # Don't compute for input layer
                d_loss_d_aL = np.dot(self.weights[l].T, delta_L)

            dW_L = np.dot(delta_L, activations[l-1].T) / m
            dB_L = np.sum(delta_L, axis=1, keepdims=True).reshape(-1,1) / m

            grads["dW"][l] = dW_L
            grads["dB"][l] = dB_L

        return grads

    def update_weights_biases(self, grad):
        for l in range(1, len(self.layers)):
            self.weights[l] = self.weights[l] - self.learning_rate * grad["dW"][l]
            self.biases[l] = self.biases[l] - self.learning_rate * grad["dB"][l]

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

