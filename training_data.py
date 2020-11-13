class TrainingSample:
    def __init__(self, inputs, outputs):
        """
        Training sample.
        :param inputs: Sequence of numbers.
        :param outputs: Sequence of numbers.
        """
        self.inputs = inputs
        self.outputs = outputs

    def __str__(self):
        return 'Training sample, inputs: {}, outputs: {}'.format(self.inputs, self.outputs)
