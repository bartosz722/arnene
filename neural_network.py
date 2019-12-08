from neuron import Neuron
import random


class NeuralNetwork:
    def __init__(self):
        """Create an uninitialized neuron."""

        self.input_count = 0

        # Each element describes one layer.
        # Each element is a list of neurons.
        # It does not include the input layer.
        # It includes the output layer.
        self.layers = []

    def initialize(self, structure, activation_function, weight_min, weight_max):
        """
        Initialize the network.
        :param structure: List of ints. The first element is the input count; the last element is
        the output count. Optional elements in the middle describe the element counts of
        the hidden layers.
        :param activation_function:
        :param weight_min:
        :param weight_max:
        :return: None
        """
        assert self.layers == []  # Check if not initialized.
        assert len(structure) >= 2
        assert all(type(x) is int and x >= 1 for x in structure)

        self.input_count = structure[0]
        prev_layer_size = self.input_count
        for curr_layer_size in structure[1:]:
            neurons_in_layer = []
            for _ in range(curr_layer_size):
                neuron = Neuron()
                weights = _get_random_float(weight_min, weight_max, prev_layer_size + 1)
                neuron.set_weights(weights[:-1], weights[-1])
                neuron.set_activation_function(activation_function)
                neurons_in_layer.append(neuron)
            self.layers.append(neurons_in_layer)
            prev_layer_size = curr_layer_size

    def calculate_outputs(self, inputs):
        assert len(inputs) == self.input_count
        curr_inputs = inputs
        for curr_layer in self.layers:
            curr_outputs = []
            for curr_neuron in curr_layer:
                neuron_output = curr_neuron.calculate_output(curr_inputs)
                curr_outputs.append(neuron_output)
            curr_inputs = curr_outputs
        return curr_inputs


def _get_random_float(min_value, max_value, value_count):
    return [random.uniform(min_value, max_value) for _ in range(value_count)]
