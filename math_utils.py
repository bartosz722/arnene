def seq_subtr(s1, s2):
    """
    Subtraction of the values from two sequences: s1 - s2.
    :param s1: Sequence 1.
    :param s2: Sequence 2.
    :return: List with the result.
    """
    assert len(s1) == len(s2)
    return [v1 - v2 for v1, v2 in zip(s1, s2)]
