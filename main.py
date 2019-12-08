import neuron
import activation_functions
import neural_network


# n = neuron.Neuron()
# n.set_activation_function(activation_functions.identity)
# n.set_weights([1, 2.2, 3.3], 0)
# out = n.calculate_output([2, 2.5, -2])
# print(out)

nn = neural_network.NeuralNetwork()
nn.initialize([4, 3, 2, 1], activation_functions.arc_tan, -1, 1)
print(nn)
out = nn.calculate_outputs((4, 2, -3, 1.5))
print(out)

