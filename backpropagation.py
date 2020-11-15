from neural_network import NeuralNetwork
from neuron import Neuron
from training_data import TrainingSample
import math_utils as mu
import log


def learning_step(network: NeuralNetwork, sample: TrainingSample):
    net_out = network.calculate_outputs(sample.inputs)
    prev_diff = mu.seq_subtr(sample.outputs, net_out)

    for i in reversed(range(len(network.layers))):
        curr_layer = network.layers[i]
        diff = prev_diff
        log.debug('learning_step(): layer {} diff: {}'.format(i, diff))
        if i >= 1:
            prev_diff = _get_diff_for_prev_layer(diff, curr_layer)
        _adjust_weights(curr_layer, diff)


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


def _adjust_weights(layer, diff):
    pass  # todo
