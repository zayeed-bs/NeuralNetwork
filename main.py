import numpy as np
from NeuralNetwork import NeuralNetwork  # Import your neural network class

# Generate training data
num_samples = 500
input_features = 2
output_features = 2

X_train = np.random.rand(num_samples, input_features)
Y_train = np.random.randint(0, 2, size=(num_samples, output_features))

# Initialize and train the neural network
nn = NeuralNetwork([2, 6, 2], learning_rate=0.01)
nn.train(X_train, Y_train, epochs=50, mini_batch_size=8)

