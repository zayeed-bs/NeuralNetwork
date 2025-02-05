import NeuralNetwork
import numpy as np

nn = NeuralNetwork.NeuralNetwork([2,15,1], 1)
activations, zs = nn.forward_propagation(np.array([0.8, 0.23]))
grad = nn.backward_propagation({0: [23,2.3]}, np.array([0.555]), activations, zs)
nn.update_weights_biases(grad)