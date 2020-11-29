from typing import List
from neural_network import NeuralNetwork
from neuron import Neuron
from training_data import TrainingSample
import math_utils as mu
import log


# noinspection PyTypeChecker
def learning_step(network: NeuralNetwork, sample: TrainingSample, eta: float):
    outputs: List[List[float]] = network.calculate_internal_outputs(sample.inputs)  # todo: [layer][out_number]
    diffs: List[List[float]] = [None] * len(network.layers)
    diffs[len(network.layers) - 1] = mu.seq_subtr(sample.outputs, outputs[len(network.layers) - 1])

    # Move backward and calculate differences.
    for i in reversed(range(1, len(network.layers))):
        diffs[i - 1] = _get_diff_for_prev_layer(diffs[i], network.layers[i])
    for i in range(len(network.layers)):
        log.debug('learning_step(): layer {} diff: {}'.format(i, diffs[i]))

    # Move forward.
    inputs = [list(sample.inputs)] + outputs[:-1]
    for i in range(len(network.layers)):
        _adjust_weights(network.layers[i], inputs[i], diffs[i], eta)


def _get_diff_for_prev_layer(diff, layer):
    """
    Calculate difference for the previous layer.
    :param diff: Difference of the current layer.
    :param layer: Current layer.
    :return: List.
    """
    assert len(diff) == len(layer)
    ret = [0.0] * len(layer[0].weights)

    for ni in range(len(layer)):
        neuron = layer[ni]
        neuron_diff = diff[ni]
        for wi in range(len(neuron.weights)):
            ret[wi] += neuron_diff * neuron.weights[wi]

    return ret


def _adjust_weights(layer: List[Neuron], layer_input: List[float], layer_diff: List[float], eta: float):
    for neuron, diff in zip(layer, layer_diff):
        for wi in range(len(neuron.weights)):
            neuron.weights[wi] += eta * neuron.activation_function_derivative(layer_input[wi])

