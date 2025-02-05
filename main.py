import NeuralNetwork
import numpy as np

nn = NeuralNetwork.NeuralNetwork([1,2,1], 0.1)
activations, zs = nn.forward_propagation(np.array([0.8]))
print(nn.backward_propagation({0: [23,2.3]}, np.array([0.555]), activations, zs))