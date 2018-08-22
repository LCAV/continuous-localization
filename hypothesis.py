import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
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
    n_anchors = 3
    n_positions = 50
    n_repetitions = 1000
    directory = "results/ranks_prob/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    ratio = n_dimensions * n_constrains / (n_dimensions * n_positions)
    max_prob = min(2 * ratio, 1)
    probabilities = np.linspace(0, max_prob, 100)

    key = "_d{}_c{}_a{}_p{}".format(n_dimensions, n_constrains, n_anchors, n_positions)

    ranks = np.zeros((len(probabilities), n_repetitions))
    # edges = np.zeros_like(ranks)
    # v_anchors = np.zeros_like(ranks)
    # v_positions = np.zeros_like(ranks)
    anchors = get_anchors(n_anchors, n_dimensions)
    frame = get_frame(n_constrains, n_positions)
    # for n_measurements in range(2, n_positions * n_anchors + 1):
    for idx, prob in enumerate(probabilities):
        for r in range(n_repetitions):
            matrix = np.random.binomial(n=1, p=prob, size=(n_anchors, n_positions))
            idx_a, idx_f = matrix_to_indexes(matrix)
            constrains = get_upper_right_constrains(idx_a, idx_f, anchors, frame)
            ranks[idx, r] = np.linalg.matrix_rank(constrains)
            # edges[idx, r] = np.sum(np.sum(matrix))
            # v_anchors[idx, r] = np.sum(np.sum(matrix, 1) >= n_constrains)
            # v_positions[idx, r] = np.sum(np.sum(matrix, 0) >= n_dimensions)

    np.save(directory + "ranks" + key, ranks)
    # np.save(directory + "anchors_degree" + key, v_anchors)
    # np.save(directory + "positions_degree" + key, v_positions)

    # DK_multiples = (np.arange(n_positions * n_anchors) + 1) / (n_dimensions * n_constrains)
    f, ax = plt.subplots()
    plt.plot(
        probabilities / ratio,
        np.mean(ranks, axis=1) / (n_dimensions * n_constrains),
        label="mean rank / full rank")
    plt.plot(
        probabilities / ratio,
        np.sum(ranks >= n_dimensions * n_constrains, axis=1) / n_repetitions,
        label="prob. of full rank")
    # plt.plot(probabilities / ratio,
    #     np.mean(edges, axis=1) / (n_anchors * n_positions))
    plt.xlabel("alpha")
    plt.grid()
    # ax.xaxis.set_major_formatter(tck.FormatStrFormatter('%g DK'))
    # ax.xaxis.set_major_locator(tck.MultipleLocator(base=1))
    plt.legend()
    plt.savefig(directory + "ranks_DK" + key + ".pdf")
    plt.show()

    # f, ax = plt.subplots()
    # plt.plot(DK_multiples, np.mean(v_anchors/n_dimensions, axis=1), label="# highly connected anchors")
    # plt.plot(DK_multiples, np.mean(v_positions/n_constrains, axis=1), label="# highly connected positions")
    # plt.xlabel("number of measurements")
    # plt.grid()
    # ax.xaxis.set_major_formatter(tck.FormatStrFormatter('%g DK'))
    # ax.xaxis.set_major_locator(tck.MultipleLocator(base=1))
    # plt.legend()
    # plt.show()
