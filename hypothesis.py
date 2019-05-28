import numpy as np
import scipy.special as special


def get_anchors(n_anchors, n_dimensions=2, scale=10):
    return scale * np.random.rand(n_dimensions, n_anchors)


def get_frame(n_constrains, n_positions):
    Ks = np.arange(n_constrains).reshape((n_constrains, 1))
    Ns = np.arange(n_positions).reshape((n_positions, 1))
    return np.cos(Ks @ Ns.T * np.pi / n_positions)


def get_left_submatrix(idx_a, idx_f, anchors, frame):
    f_vect = [frame[:, i] for i in idx_f]
    a_extended = [np.concatenate([anchors[:, a], [1]]) for a in idx_a]
    matrices = [(a[:, None] @ f[None, :]).flatten() for (f, a) in zip(f_vect, a_extended)]
    return np.array(matrices).T


def get_reduced_right_submatrix(idx_f, frame):
    vectors = [frame[:, idx] for idx in idx_f]
    matrix = [f[1:] * f[-1] for f in vectors]
    return np.array(matrix).T


def random_indexes(n_anchors, n_positions, n_measurements):
    assert n_positions * n_anchors >= n_measurements, "to many measurements requested"
    indexes = np.random.choice(n_positions * n_anchors, n_measurements, replace=False)
    idx_a, idx_f = np.unravel_index(indexes, (n_anchors, n_positions))
    return idx_a.tolist(), idx_f.tolist()


def indexes_to_matrix(idx_a, idx_f, n_anchors, n_positions):
    matrix = np.zeros((n_anchors, n_positions))
    matrix[idx_a, idx_f] = 1
    return matrix


def matrix_to_indexes(matrix):
    indexes = np.argwhere(matrix)
    return indexes[:, 0].tolist(), indexes[:, 1].tolist()


def probability_few_anchors(n_dimensions, n_constrains, n_positions):
    """Calculate probability of size the smallest matrix being invertible with minimal number of anchors.

    Smallest matrix that can invertible has to have (n_dimensions+1)n_constrains rows,
    and the smallest number of needed anchors is n_dimensions + 1. The behaviour is qualitatively different
    for n_dimensions + 1 anchors than for more anchors."""
    total = special.binom((n_dimensions + 1) * n_positions, (n_dimensions + 1) * n_constrains)
    full = special.binom(n_positions, n_constrains)**(n_dimensions + 1)
    return full / total


def probability_few_anchors_limit(n_dimensions, n_constrains):
    """Calculate analytical limit of probability_few_anchors"""
    return np.sqrt(n_dimensions + 1) / (np.sqrt(2 * np.pi * n_constrains)**n_dimensions)
