import log
from training_data import TrainingSample


def load_f01(file_path):
    """
    Load training data from a format 01 file.
    :return: Sequence[TrainingSample]
    """
    log.info("Load training data format 01 from: " + file_path)
    ret = []

    with open(file_path) as f:
        header = f.readline().split(',')
        dim = int(header[0])  # letters have dim X dim pixels
        out_count = int(header[1])  # letter types count
        log.debug("dim: {}, out_count: {}".format(dim, out_count))

        curr_sample = None
        curr_sample_row = 0

        for line in f:
            line = line.rstrip()
            if not line or line[0] == '#':
                continue  # ignore empty lines and comments
            if not curr_sample:
                curr_sample = TrainingSample([0] * dim**2, [])
                curr_sample_row = 0

            if curr_sample_row < dim:  # letter image
                weight_idx = curr_sample_row * dim
                assert len(line) <= dim, 'Too many characters in a letter\'s row: ' + line
                for i, char in enumerate(line):
                    if char != ' ':
                        curr_sample.inputs[weight_idx + i] = 1
                curr_sample_row += 1
            else:  # output values
                curr_sample.outputs = [float(x) for x in line.split(',')]
                assert len(curr_sample.outputs) == out_count, 'Invalid output count in line: ' + line
                ret.append(curr_sample)
                log.debug('Adding sample: {}'.format(curr_sample))
                curr_sample = None

    log.info('Loaded {} samples'.format(len(ret)))
    return ret


def load_f02(file_path):
    """
    Load training data from a format 02 file.
    :return: Sequence[TrainingSample]
    """
    log.info("Load training data format 02 from: " + file_path)
    ret = []

    with open(file_path) as f:
        number_of_inputs = len(f.readline().split(','))
        number_of_outputs = len(f.readline().split(','))
        assert number_of_inputs >= 1
        assert number_of_outputs >= 1

        sample_pointer = 0  # 0, 1, 2

        for line in f:
            line = line.strip()
            if not line or line[0] == '#':
                continue  # ignore empty lines and comments

            if sample_pointer == 0:
                sample_name = line
            elif sample_pointer == 1:
                inputs = [float(x) for x in line.split(',')]
                assert len(inputs) == number_of_inputs, \
                    'Invalid input count in sample {}: {}'.format(sample_name, len(inputs))
            else:
                outputs = [float(x) for x in line.split(',')]
                assert len(outputs) == number_of_outputs, \
                    'Invalid output count in sample {}: {}'.format(sample_name, len(outputs))
                sample = TrainingSample(inputs, outputs)
                log.debug('Adding sample: {}, {}'.format(sample_name, sample))
                ret.append(sample)

            sample_pointer += 1
            if sample_pointer > 2:
                sample_pointer = 0

    assert sample_pointer == 0, 'The input file is unfinished'
    log.info('Loaded {} samples'.format(len(ret)))
    return ret
