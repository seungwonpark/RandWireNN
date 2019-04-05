import numpy as np


def shuffle(n, edges):
    mapping = np.random.permutation(range(n))
    shuffled = list()
    for edge in edges:
        s, e = edge
        shuffled.append(sorted((mapping[s], mapping[e])))
    return sorted(shuffled)
