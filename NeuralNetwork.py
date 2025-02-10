import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1/(1 + np.exp(-z))

def der_sigmoid(z):
    return sigmoid(z) * (1-sigmoid(z))

def relu(z):
    return np.maximum(0, z)

def der_relu(z):
    return np.where(z>0, 1, 0)

class NeuralNetwork:
    def forward_propagation(self, X):
        activations = {0: X}
        zs = {}

        for l in range(1, len(self.layers)):
            z = np.dot(self.weights[l], activations[l-1]) + self.biases[l]
            zs[l] = z

            if l == len(self.layers)-1:  # Output layer
                activations[l] = sigmoid(z)  # For classification
            else:  # Hidden layers
                activations[l] = relu(z)

        y_pred = activations[len(self.layers) - 1]

        return y_pred, activations, zs

    def calculate_loss(self, Y, Y_pred):
        Y = Y.reshape(self.layers[-1], -1)
        return np.mean((Y-Y_pred)**2)

    def backward_propagation(self, X, Y, activations:dict, zs):
        Y = Y.reshape(self.layers[-1], -1)
        L = len(self.layers)-1
        m = X.shape[1] if len(X.shape) > 1 else 1
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


    def train(self, X, Y, epochs=100, mini_batch_size=0):
        dat_size = X.shape[0]

        if Y.shape[0] != dat_size:
            return 0

        debug_loss = []

        for e in range(epochs):
            # Shuffle dataset
            shuffle_ind = np.arange(dat_size)
            np.random.shuffle(shuffle_ind)

            Y_pred_list = []  # Store predictions across all batches

            # Check if there are mini batches
            if mini_batch_size:
                shuffle_ind = np.array_split(shuffle_ind, dat_size // mini_batch_size)

                for index in shuffle_ind:
                    # Extract mini-batch and reshape correctly
                    x = np.array([X[i] for i in index]).T
                    y = np.array([Y[i] for i in index]).reshape(1, -1)  # Ensure (1, batch_size)

                    # Forward propagation
                    Y_pred_batch, activations, zs = self.forward_propagation(x)
                    Y_pred_list.append(Y_pred_batch)

                    # Backward propagation using Y_batch, NOT Y_pred
                    grads = self.backward_propagation(x, y, activations, zs)
                    self.update_weights_biases(grads)

                # Convert collected predictions to a NumPy array
                Y_pred = np.concatenate(Y_pred_list, axis=1)

            else:  # Full-batch mode
                for index in shuffle_ind:
                    y, activations, zs = self.forward_propagation(X[index].reshape(-1, 1))
                    Y_pred_list.append(y)

                    x = np.atleast_2d(X[index]).T
                    grads = self.backward_propagation(x, y, activations, zs)
                    self.update_weights_biases(grads)

                Y_pred = np.vstack(Y_pred_list)

            # Compute and store loss for the epoch
            loss = self.calculate_loss(Y.reshape(self.layers[-1], -1), Y_pred)
            print(f"Epoch {e + 1}, Loss: {loss}")
            debug_loss.append(loss)

        plt.plot(debug_loss)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Time")
        plt.show()

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

