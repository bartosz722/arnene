from neuron import Neuron
import random


class NeuralNetwork:
    def __init__(self):
        """Create an uninitialized neural network."""

        self.input_count = 0

        # Each element describes one layer.
        # Each element is a list of neurons.
        # It does not include the input layer.
        # It includes the output layer.
        self.layers = []

    def __str__(self):
        ret = 'Neural network, '
        for layer in self.layers:
            ret += 'Layer, '
            for n in layer:
                ret += str(n) + ', '
        return ret

    def initialize(self, input_count, neuron_counts, activation_function,
                   activation_function_derivative, weight_min, weight_max):
        """
        Initialize the network.
        :param input_count: Number of inputs.
        :param neuron_counts: List of ints. It tells how many neurons are in each layer;
        this does not include the input layer. The last element also defines the output count.
        :param activation_function:
        :param activation_function_derivative: May be None.
        :param weight_min:
        :param weight_max:
        :return: None
        """
        assert self.layers == []  # Check if not initialized.
        assert len(neuron_counts) >= 1
        assert all(type(x) is int and x >= 1 for x in neuron_counts)

        prev_layer_size = self.input_count = input_count
        for curr_layer_size in neuron_counts:
            neurons_in_layer = []
            for _ in range(curr_layer_size):
                neuron = Neuron()
                weights = _get_random_float(weight_min, weight_max, prev_layer_size + 1)
                neuron.set_weights(weights[:-1], weights[-1])
                neuron.set_activation_function(activation_function, activation_function_derivative)
                neurons_in_layer.append(neuron)
            self.layers.append(neurons_in_layer)
            prev_layer_size = curr_layer_size

    def calculate_outputs(self, inputs):
        """
        Calculate the outputs of the network.
        :param inputs: Sequence of numbers
        :return: List of numbers.
        """
        assert len(inputs) == self.input_count
        curr_inputs = inputs
        for curr_layer in self.layers:
            curr_inputs = self._calculate_layer_outputs(curr_layer, curr_inputs)
        return curr_inputs

    def calculate_intermediate_outputs(self, inputs):
        """
        Calculate outputs of each layer of the network.
        :param inputs: Sequence of numbers
        :return: List of lists of numbers. Indexes: [layer][output].
        """
        assert len(inputs) == self.input_count
        ret = []
        curr_inputs = inputs
        for curr_layer in self.layers:
            curr_inputs = self._calculate_layer_outputs(curr_layer, curr_inputs)
            ret.append(curr_inputs)
        return ret

    @staticmethod
    def _calculate_layer_outputs(layer, inputs):
        return [neuron.calculate_output(inputs) for neuron in layer]

    def get_neuron(self, layer_number, neuron_number):
        return self.layers[layer_number][neuron_number]


def _get_random_float(min_value, max_value, value_count):
    return [random.uniform(min_value, max_value) for _ in range(value_count)]
