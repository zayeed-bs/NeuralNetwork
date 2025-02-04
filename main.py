import NeuralNetwork

nn = NeuralNetwork.NeuralNetwork([1,1,1], 0.1)
print(nn.forward_propagation(2))