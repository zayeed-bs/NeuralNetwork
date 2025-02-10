import NeuralNetwork
import numpy as np

nn = NeuralNetwork.NeuralNetwork([1,2,1], 0.01)

X_train = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])  # Inputs
Y_train = 2 * X_train + np.random.randn(5, 1) * 0.1  # Outputs (y = 2x + noise)


nn.train(X_train, Y_train, epochs=1000, mini_batch_size=2)
