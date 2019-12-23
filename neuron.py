class Neuron:
    """Neuron."""

    def __init__(self):
        """Create an uninitialized neuron."""
        self.weights = []
        self.bias = 0
        self.activation_function = None

    def __str__(self):
        return 'Neuron, bias: {}, weights: {}'.format(self.bias, self.weights)

    def set_weights(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def set_activation_function(self, function):
        self.activation_function = function

    @property
    def input_count(self):
        return len(self.weights)

    def calculate_output(self, inputs):
        """
        Calculate the output of the neuron.
        :param inputs: Sequence[float]
        :return: float
        """
        assert len(inputs) == self.input_count
        sumx = sum(i * w for i, w in zip(inputs, self.weights))
        return float(self.activation_function(sumx + self.bias))

