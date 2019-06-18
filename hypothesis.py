import numpy as np
import scipy.special as special
import time


def get_anchors(n_anchors, n_dimensions=2, scale=10):
    return scale * np.random.rand(n_dimensions, n_anchors)


def get_frame(n_constrains, n_positions):
    Ks = np.arange(n_constrains).reshape((n_constrains, 1))
    Ns = np.arange(n_positions).reshape((n_positions, 1))
    return np.cos(Ks @ Ns.T * np.pi / n_positions)


def get_left_submatrix(idx_a, idx_f, anchors, frame):
    """Generated left submatrix in the extended form.

    The extended form means that anchors are extended by 1 making them n_dimensions+1 dimensional.
    Trows of the matrix are tensor product of extended anchors and frame vectors,
    which means that the whole matrix is (n_dimensions+1) * n_constrains x n_measurements dimensional.
    This is because it seems to be a more natural representation - for localising just one point,
    the full matrix is reducted to the (extended) left submatrix.

    :param idx_a: list of anchor indexes for each measurement
    :param idx_f: list of frame indexes for each measurement
    :param anchors: matrix of all available anchors, of size n_dimensions x n_anchors
    :param frame: matrix of all frame vectors, of size n_points x n_constrains

    :return: left part of the constrain matrix
    """

    f_vect = [frame[:, i] for i in idx_f]
    a_extended = [np.concatenate([anchors[:, a], [1]]) for a in idx_a]
    matrices = [(a[:, None] @ f[None, :]).flatten() for (f, a) in zip(f_vect, a_extended)]
    return np.array(matrices).T


def get_reduced_right_submatrix(idx_f, frame):
    """Generated right submatrix in the reduced form.

    The extended form means the tensor products of frame vectors are reduced to a basis,
    which means that the size of the submatrix is (n_constrains - 1) x n_measurements.

    :param idx_f: list of frame indexes for each measurement
    :param frame: matrix of all frame vectors, of size n_points x n_constrains

    :return: right part of the constrain matrix
    """
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


def probability_few_anchors_limit(n_dimensions, n_constrains, anchors_limit=False):
    """Calculate analytical limit of probability_few_anchors.

    Based on known binomial symbol limits for minimum number of anchors, fixed number of constrains and large number of
    positions.
    Can be also used to calculate limit for minimum number of positions, fixed number of dimensions and large number
    of anchors.

    :param n_dimensions: number of dimensions D
    :param n_constrains: number of constrains K
    :param anchors_limit: if true, calculate limit for many anchors, not for many measurements
    :return:
        float: a limit of probability of the left hand side matrix being full rank, as number of positions (or number of
        constrains) goes to infinity
    """
    if anchors_limit:
        (n_dimensions, n_constrains) = (n_constrains - 1, n_dimensions + 1)

    return np.sqrt(n_dimensions + 1) / (np.sqrt(2 * np.pi * n_constrains)**n_dimensions)


def probability_upper_bound(n_dimensions, n_constrains, n_positions, n_anchors, position_wise=False):
    """Calculate upper bound on the probability of matrix being full rank,
    assuming that the number of measurements is exactly n_constrains * (n_dimensions + 1).
    This assumption allows for speed up calculations.

    :param n_dimensions: number of dimensions D
    :param n_constrains: number of constrains K
    :param n_positions: number of positions along trajectory N
    :param n_anchors: number of anchors M
    :param position_wise: if True, calculates the upper bound based on partitions of positions and not anchors.
    The equations are the same for both cases, but for readability purposes the naming convention for partition of
    anchors is used rather than abstract notation
    :return:
        float: upper bound on probability of the left hand side of the matrix being full rank
    """

    if position_wise:
        (n_dimensions, n_constrains) = (n_constrains - 1, n_dimensions + 1)
        (n_anchors, n_positions) = (n_positions, n_anchors)

    start = time.time()

    max_index = n_constrains + 1
    n_measurements = n_constrains * (n_dimensions + 1)
    upper_bound_combinations = 0

    for part_nr in range(max_index**n_anchors):
        partition = np.unravel_index(part_nr, [max_index] * n_anchors)
        if sum(partition) == n_measurements:  # go through all partition of measurements between anchors
            new_combinations = 1
            for k in partition:
                new_combinations *= special.binom(n_positions, k)
            upper_bound_combinations += new_combinations

    total_combinations = special.binom(n_positions * n_anchors, n_measurements)
    end = time.time()
    print("Upper bound, position {}, elapsed time: {:.2f}s".format(position_wise, end - start))
    return upper_bound_combinations / total_combinations


# TODO from the plots this lower bound does not seem to be lower.
# It seems to calculate the upper bound minus a constant?
def probability_lower_bound(n_dimensions, n_constrains, n_positions, n_anchors):

    start = time.time()
    max_index = n_constrains + 1
    n_measurements = n_constrains * (n_dimensions + 1)
    upper_bound_combinations = 0

    bad_combinations = 0
    for part_nr in range(max_index**n_anchors):
        partition = np.unravel_index(part_nr, [max_index] * n_anchors)
        if sum(partition) == n_measurements:  # go through all partition of measurements between anchors
            for subset_nr in range(2**n_anchors):
                subset = np.unravel_index(subset_nr, [2] * n_anchors)
                if sum(subset) == n_dimensions + 2:  # look for size D+2 clashes
                    new_bad_combinations = n_positions
                    for k, i in zip(partition, subset):
                        new_bad_combinations *= special.binom(n_positions, k - i)
                    bad_combinations += new_bad_combinations

            new_combinations = 1
            for k in partition:
                new_combinations *= special.binom(n_positions, k)
            upper_bound_combinations += new_combinations

    end = time.time()
    print("lower time: {:.2f}s".format(end - start))
    total_combinations = special.binom(n_positions * n_anchors, n_measurements)
    return (upper_bound_combinations - bad_combinations) / total_combinations


def left_independence_estimation(n_constrains, min_anchors, poisson_mean, repetitions=10000, use_limits=False):
    """
    Estimate joint and marginal probabilities of the left hand side matrix satisfying
    certain anchor and positions conditions necessary for the matrix to be full rank


    :param n_constrains: number of constrains K
    :param min_anchors: minimum number of anchors (D+1)
    :param poisson_mean: mean of the poisson variable added n_constrains
    and min_anchors to obtain number of positions and number of anchors
    :param repetitions: number of times to generate data
    :param use_limits: if True uses new limit conditions, otherwise use two out of four old conditions
    :return:
        tuple (number of good row configurations, number of good row and column configurations,
        number of entirely good row configurations, total number of tests made
    """

    def limit_condition(m, a, b, axis):
        row_sum = np.sort(np.sum(m, axis=axis))
        extra = row_sum[:-a]
        missing = np.clip(b - row_sum[-a:], a_min=0, a_max=None)
        return np.sum(extra) >= np.sum(missing)

    anchors_ok = 0
    positions_ok = 0
    both_ok = 0
    total = 0
    for _ in range(repetitions):
        n_anchors = min_anchors + np.random.poisson(poisson_mean)
        n_positions = n_constrains + np.random.poisson(poisson_mean)
        p = min(1, (min_anchors * n_constrains) / (n_anchors * n_positions))
        matrix = np.random.binomial(1, p, size=(n_positions, n_anchors))
        # Proceed only if we have enough measurements
        if np.sum(matrix) >= n_constrains * min_anchors:
            total += 1
            positions_feasible = True
            anchors_feasible = True
            if use_limits:
                anchors_feasible = limit_condition(matrix, min_anchors, n_constrains, axis=0)
                positions_feasible = limit_condition(matrix, n_constrains, min_anchors, axis=1)
            else:
                for part_nr in range(2**n_positions):
                    partition = np.unravel_index(part_nr, [2] * n_positions)
                    if sum(partition) == (n_positions - (n_constrains - 1)):
                        # it sets feasible to false!
                        if np.sum(np.multiply(partition, np.sum(matrix, axis=1))) < min_anchors:
                            positions_feasible = False
                            break
                for part_nr in range(2**n_anchors):
                    partition = np.unravel_index(part_nr, [2] * n_anchors)
                    if sum(partition) == (n_anchors - (min_anchors - 1)):
                        # it sets feasible to false!
                        if np.sum(np.multiply(partition, np.sum(matrix, axis=0))) < n_constrains:
                            anchors_feasible = False
                            break

            if anchors_feasible:
                anchors_ok += 1
            if positions_feasible:
                positions_ok += 1
            if anchors_feasible and positions_feasible:
                both_ok += 1

    return anchors_ok, positions_ok, both_ok, total
