from neural_network import NeuralNetwork
import log


def create_perceptron(input_count, activation_function, weight_min, weight_max):
    """Create a neural network in the form of a perceptron."""
    p = NeuralNetwork()
    p.initialize(input_count, [1], activation_function, weight_min, weight_max)
    return p


# TODO: random sample selection
def learn(perceptron, training_samples, learning_rate, learning_iterations, error_threshold):
    """
    Simple perceptron learning algorithm, SPLA.
    It stops after the specified number of iterations, or if the error threshold has been reached.
    :param perceptron: Perceptron
    :param training_samples: Sequence[TrainingSample]
    :param learning_rate:
    :param learning_iterations:
    :param error_threshold:
    :return: None
    """
    log.info("Learning started")
    neuron = perceptron.get_neuron(0, 0)

    for learn_i in range(learning_iterations):
        log.info("Iteration {}".format(learn_i + 1))

        total_error_abs = 0
        for sample_i in range(len(training_samples)):
            log.info("Sample {}".format(sample_i))

            sample = training_samples[sample_i]
            current_output = perceptron.calculate_outputs(sample.inputs)[0]
            wanted_output = sample.outputs[0]
            output_error = wanted_output - current_output
            total_error_abs += abs(output_error)
            log.info("Output error: {}".format(output_error))

            weight_change = learning_rate * output_error
            for i in range(len(neuron.weights)):
                neuron.weights[i] += weight_change * sample.inputs[i]
            neuron.bias += weight_change

        mean_error = total_error_abs / float(len(training_samples))
        log.info("Mean error in this iteration: {}".format(mean_error))
        if mean_error <= error_threshold:
            log.info("Reached the error threshold after iteration {}, stopping.".format(learn_i + 1))
            return

    log.info("Done all {} learning iterations, stopping.".format(learning_iterations))

