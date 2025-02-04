import NeuralNetwork
import numpy as np

nn = NeuralNetwork.NeuralNetwork([4,4,4], 0.1)
print(nn.forward_propagation(np.array([0.23,0.12,0.22,0.51])))