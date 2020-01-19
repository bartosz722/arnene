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
    p = perceptron.create_perceptrons(
        2,
        1,
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
    pp = perceptron.create_perceptrons(
        5*5,
        4,
        activation_functions.binary_step_unipolar,
        -1,
        1)
    training_data = training_data_loader.load_f01('training-data/f01-letters-1.txt')

    print(pp)
    perceptron.learn(pp, training_data, 0.1, 1000, 0.001)
    print(pp)

    testing_data = training_data_loader.load_f01('training-data/f01-letters-1-test.txt')
    perceptron.test(pp, testing_data, 0.1)


def main4():
    pp = perceptron.create_perceptrons(
        5,
        3,
        activation_functions.binary_step_bipolar,
        -1,
        1)
    training_data = training_data_loader.load_f02('training-data/f02-animals.txt')

    print(pp)
    perceptron.learn(pp, training_data, 0.1, 1000, 0.001)
    print(pp)

    testing_data = training_data_loader.load_f02('training-data/f02-animals-test.txt')
    perceptron.test(pp, testing_data, 0.1)


main4()
