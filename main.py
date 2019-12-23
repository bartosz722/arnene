import neuron
import activation_functions
import neural_network
import perceptron
import log
from training_data import TrainingSample
import training_data_loader


def main1():
    n = neuron.Neuron()
    n.set_activation_function(activation_functions.identity)
    n.set_weights([1, 2.2, 3.3], 0)
    out = n.calculate_output([2, 2.5, -2])
    print(out)

    nn = neural_network.NeuralNetwork()
    nn.initialize(4, [3, 2, 1], activation_functions.arc_tan, -1, 1)
    print(nn)
    out = nn.calculate_outputs((4, 2, -3, 1.5))
    print(out)


def main2():
    p = perceptron.create_perceptron(
        2,
        activation_functions.binary_step_unipolar,
        -1,
        1)
    td = (
        TrainingSample([2, 4], [1]),
        TrainingSample([2, -3], [0]),
        TrainingSample([6, -1], [0]),
        TrainingSample([6, 4], [1]),
        TrainingSample([3, 2], [1]),
        TrainingSample([8, 2], [0]),
    )

    print(p)
    perceptron.learn(p, td, 0.1, 1000, 0.001)
    print(p)


def main3():
    p = perceptron.create_perceptron(
        5*5,
        activation_functions.binary_step_unipolar,
        -1,
        1)
    td = training_data_loader.load_f01('training-data/f01-letters-1.txt')

    # TODO: multi-perceptron
    print(p)
    letter_idx = 0
    tdx = [TrainingSample(x.inputs, [x.outputs[letter_idx]]) for x in td]
    perceptron.learn(p, tdx, 0.1, 1000, 0.001)
    print(p)


main3()
