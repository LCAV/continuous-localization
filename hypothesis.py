import numpy as np
import os


def get_anchors(n_anchors, n_dimensions=2, scale=10):
    return scale * np.random.rand(n_dimensions, n_anchors)


def get_frame(n_constrains, n_positions):
    Ks = np.arange(n_constrains).reshape((n_constrains, 1))
    Ns = np.arange(n_positions).reshape((n_positions, 1))
    return np.cos(Ks @ Ns.T * np.pi / n_positions)


def get_full_constrains(idx_a, idx_f, anchors, frame):
    vectors = [np.append(anchors[:, a], frame[:, f]) for (a, f) in zip(idx_a, idx_f)]
    matrices = [(v[np.newaxis, :].T @ v[np.newaxis, :]).flatten() for v in vectors]
    return np.array(matrices).T


def get_upper_right_constrains(idx_a, idx_f, anchors, frame):
    matrices = [(anchors[:, a:a + 1] @ frame[:, f:f + 1].T).flatten()
                for (a, f) in zip(idx_a, idx_f)]
    return np.array(matrices).T


def random_indexes(n_anchors, n_positions, n_measurements):
    assert n_positions * n_anchors >= n_measurements, "to many measurements requested"
    indexes = np.random.choice(n_positions * n_anchors, n_measurements, replace=False)
    idx_a, idx_f = np.unravel_index(indexes, (n_anchors, n_positions))
    return (idx_a.tolist(), idx_f.tolist())


def indexes_to_matrix(idx_a, idx_f, n_anchors, n_positions):
    matrix = np.zeros((n_anchors, n_positions))
    matrix[idx_a, idx_f] = 1
    return matrix


def matrix_to_indexes(matrix):
    indexes = np.argwhere(matrix)
    return (indexes[:, 0].tolist(), indexes[:, 1].tolist())


if __name__ == '__main__':

    n_dimensions = 2
    n_constrains = 5
    n_anchors = 10
    n_positions = 30
    n_repetitions = 10
    fixed_probability = True
    directory = "results/ranks/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    probabilities = np.linspace(0, 1, 100)

    key = "_d{}_c{}_a{}_p{}_{}".format(n_dimensions, n_constrains, n_anchors, n_positions,
                                       fixed_probability)

    anchors = get_anchors(n_anchors, n_dimensions)
    frame = get_frame(n_constrains, n_positions)

    if fixed_probability:
        ranks = np.zeros((len(probabilities), n_repetitions))
        for idx, prob in enumerate(probabilities):
            for r in range(n_repetitions):
                matrix = np.random.binomial(n=1, p=prob, size=(n_anchors, n_positions))
                idx_a, idx_f = matrix_to_indexes(matrix)
                constrains = get_upper_right_constrains(idx_a, idx_f, anchors, frame)
                ranks[idx, r] = np.linalg.matrix_rank(constrains)
    else:
        ranks = np.zeros((n_positions * n_anchors + 1, n_repetitions))
        for n_measurements in range(2, n_positions * n_anchors + 1):
            for r in range(n_repetitions):
                idx_a, idx_f = random_indexes(n_anchors, n_positions, n_measurements)
                constrains = get_upper_right_constrains(idx_a, idx_f, anchors, frame)
                ranks[n_measurements, r] = np.linalg.matrix_rank(constrains)

    np.save(directory + "ranks" + key, ranks)
