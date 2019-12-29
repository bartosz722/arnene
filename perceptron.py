from neural_network import NeuralNetwork
import log
import random


def create_perceptrons(input_count, output_count, activation_function, weight_min, weight_max):
    """Create a neural network with one ore more perceptrons."""
    nn = NeuralNetwork()
    nn.initialize(input_count, [output_count], activation_function, weight_min, weight_max)
    return nn


def _learn_using_sample(neuron, training_sample, output_index, learning_rate, error_threshold):
    """
    Execute one learning step on a neuron being a single perceptron.
    :param neuron: Neuron to learn.
    :param training_sample: Training sample. It may contain outputs for many neurons.
    :param output_index: Index of the output value in the training sample.
    :param learning_rate: Learning rate.
    :param error_threshold: Error threshold.
    :return: It returns the absolute value of the output error as it was before the weights
    adjustment. If this value is <= the error threshold, None is returned and the weights
    are not changed.
    """
    sample_inputs = training_sample.inputs
    sample_output = training_sample.outputs[output_index]
    perceptron_output = neuron.calculate_output(sample_inputs)
    output_error = sample_output - perceptron_output
    output_error_abs = abs(output_error)

    if output_error_abs <= error_threshold:
        return None  # no more learning

    weight_change = learning_rate * output_error
    for i in range(len(neuron.weights)):
        neuron.weights[i] += weight_change * sample_inputs[i]
    neuron.bias += weight_change

    return output_error_abs


def learn(perceptron_network, training_samples, learning_rate, max_learning_iterations,
          error_threshold):
    """
    Learn a network with one or more perceptrons.
    Simple perceptron learning algorithm (SPLA) is used.
    It stops after the specified number of iterations, or if the error threshold has been reached
    for all samples for all perceptrons.
    :param perceptron_network: A network with one or more perceptrons.
    :param training_samples: Sequence[TrainingSample]
    :param learning_rate: Learning rate.
    :param max_learning_iterations: Maximum number of learning iterations.
    :param error_threshold: Error threshold.
    :return: True if the learning succeeded, false otherwise
    """
    learning_finished = set()  # indexes of neurons
    neurons = perceptron_network.layers[0]
    sample_indexes = list(range(len(training_samples)))

    assert len(perceptron_network.layers) == 1, 'The neural network must have only one layer.'
    assert all(len(s.inputs) == perceptron_network.input_count for s in training_samples), \
        'All samples must have {} inputs.'.format(perceptron_network.input_count)
    assert all(len(s.outputs) == len(neurons) for s in training_samples), \
        'All samples must have {} outputs.'.format(len(neurons))

    log.info("Learning started")
    learning_iteration = 0
    while True:
        learning_iteration += 1
        log.info("Iteration {}".format(learning_iteration))

        all_samples_ok_for_neuron = [True] * len(neurons)
        random.shuffle(sample_indexes)  # random order of the samples

        # iterate the training samples
        for sample_index in sample_indexes:
            log.debug('Processing sample with index {}'.format(sample_index))
            sample = training_samples[sample_index]

            # iterate the neurons
            for neuron_idx, neuron in enumerate(neurons):
                if neuron_idx in learning_finished:
                    continue
                neuron_error = _learn_using_sample(
                    neuron, sample, neuron_idx, learning_rate, error_threshold)
                log.debug('Neuron {} error: {}'.format(neuron_idx, neuron_error))
                if neuron_error is not None:
                    all_samples_ok_for_neuron[neuron_idx] = False

        # Check which neurons are trained enough.
        for i, ok in enumerate(all_samples_ok_for_neuron):
            if ok and i not in learning_finished:
                log.info('Neuron {} has been trained enough.'.format(i))
                learning_finished.add(i)

        if len(learning_finished) == len(neurons):
            log.info("Learning succeeded. Reached the wanted threshold after {} iterations."
                     .format(learning_iteration))
            return True

        if learning_iteration == max_learning_iterations:
            log.info("Learning failed. Done all {} learning iterations."
                     .format(max_learning_iterations))
            return False
