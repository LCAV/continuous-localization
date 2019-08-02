import numpy as np
import scipy.special as special
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from plotting_tools import make_dirs_safe

# TODO change the diemsions so the matrices dont need to be transposed


def get_anchors(n_anchors, n_dimensions=2, check=True):
    full_rank = False
    extension = np.ones((1, n_anchors))
    anchors = np.zeros((n_dimensions, n_anchors))
    if check:
        while not full_rank:
            # TODO we would ideally like to check if any subset of anchors of the size
            #  n_dimensions+1 is full rank, but it probably does not matter in the end
            anchors = np.random.rand(n_dimensions, n_anchors)
            extended = np.concatenate([anchors, extension])
            if np.linalg.matrix_rank(extended) > n_dimensions:
                full_rank = True
    return anchors


def get_frame(n_constraints, n_positions):
    Ks = np.arange(n_constraints).reshape((n_constraints, 1))
    Ns = np.arange(n_positions).reshape((n_positions, 1))
    return np.cos(Ks @ Ns.T * np.pi / n_positions)


def get_left_submatrix(idx_a, idx_f, anchors, frame):
    """Generated left submatrix in the extended form.

    The extended form means that anchors are extended by 1 making them n_dimensions+1 dimensional.
    Trows of the matrix are tensor product of extended anchors and frame vectors,
    which means that the whole matrix is (n_dimensions+1) * n_constraints x n_measurements dimensional.
    This is because it seems to be a more natural representation - for localising just one point,
    the full matrix is reducted to the (extended) left submatrix.

    :param idx_a: list of anchor indexes for each measurement
    :param idx_f: list of frame indexes for each measurement
    :param anchors: matrix of all available anchors, of size n_dimensions x n_anchors
    :param frame: matrix of all frame vectors, of size n_points x n_constraints

    :return: left part of the constrain matrix
    """

    f_vect = [frame[:, i] for i in idx_f]
    a_extended = [np.concatenate([anchors[:, a], [1]]) for a in idx_a]
    matrices = [(a[:, None] @ f[None, :]).flatten() for (f, a) in zip(f_vect, a_extended)]
    return np.array(matrices).T


def get_reduced_right_submatrix(idx_f, frame):
    """Generated right submatrix in the reduced form.

    The extended form means the tensor products of frame vectors are reduced to a basis,
    which means that the size of the submatrix is (n_constraints - 1) x n_measurements.

    :param idx_f: list of frame indexes for each measurement
    :param frame: matrix of all frame vectors, of size n_points x n_constraints TODO

    :return: right part of the constrain matrix
    """

    n_constraints, n_positions = frame.shape
    Ks = np.arange(n_constraints, 2 * n_constraints - 1).reshape((n_constraints - 1, 1))
    Ns = np.arange(n_positions).reshape((n_positions, 1))
    extended_frame = np.cos(Ks @ Ns.T * np.pi / n_positions)
    vectors = [extended_frame[:, idx] for idx in idx_f]
    matrix = np.array(vectors)
    return np.array(matrix).T


def get_full_matrix(idx_a, idx_f, anchors, frame):
    return np.concatenate([get_left_submatrix(idx_a, idx_f, anchors, frame), get_reduced_right_submatrix(idx_f, frame)])


def random_indexes(n_anchors, n_positions, n_measurements):
    if n_positions * n_anchors < n_measurements:
        raise ValueError("to many measurements {}>{}x{}requested ".format(n_measurements, n_positions, n_anchors))
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


def probability_few_anchors(n_dimensions, n_constraints, n_positions):
    """Calculate probability of size the smallest matrix being invertible with minimal number of anchors.

    Smallest matrix that can invertible has to have (n_dimensions+1)n_constraints rows,
    and the smallest number of needed anchors is n_dimensions + 1. The behaviour is qualitatively different
    for n_dimensions + 1 anchors than for more anchors."""

    total = special.binom((n_dimensions + 1) * n_positions, (n_dimensions + 1) * n_constraints)
    full = special.binom(n_positions, n_constraints)**(n_dimensions + 1)
    return full / total


def probability_few_anchors_limit(n_dimensions, n_constraints, anchors_limit=False):
    """Calculate analytical limit of probability_few_anchors.

    Based on known binomial symbol limits for minimum number of anchors, fixed number of constrains and large number of
    positions.
    Can be also used to calculate limit for minimum number of positions, fixed number of dimensions and large number
    of anchors.

    :param n_dimensions: number of dimensions D
    :param n_constraints: number of constrains K
    :param anchors_limit: if true, calculate limit for many anchors, not for many measurements
    :return:
        float: a limit of probability of the left hand side matrix being full rank, as number of positions (or number of
        constrains) goes to infinity
    """
    if anchors_limit:
        (n_dimensions, n_constraints) = (n_constraints - 1, n_dimensions + 1)

    return np.sqrt(n_dimensions + 1) / (np.sqrt(2 * np.pi * n_constraints)**n_dimensions)


def probability_upper_bound(n_dimensions, n_constraints, n_positions, n_anchors, position_wise=False):
    """Calculate upper bound on the probability of matrix being full rank,
    assuming that the number of measurements is exactly n_constraints * (n_dimensions + 1).
    This assumption allows for speed up calculations.

    :param n_dimensions: number of dimensions D
    :param n_constraints: number of constrains K
    :param n_positions: number of positions along trajectory N
    :param n_anchors: number of anchors M
    :param position_wise: if True, calculates the upper bound based on partitions of positions and not anchors.
    The equations are the same for both cases, but for readability purposes the naming convention for partition of
    anchors is used rather than abstract notation
    :return:
        float: upper bound on probability of the left hand side of the matrix being full rank
    """

    if position_wise:
        (n_dimensions, n_constraints) = (n_constraints - 1, n_dimensions + 1)
        (n_anchors, n_positions) = (n_positions, n_anchors)

    start = time.time()

    max_index = n_constraints + 1
    n_measurements = n_constraints * (n_dimensions + 1)
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
    if end - start > 1:
        print("Upper bound, position {}, elapsed time: {:.2f}s".format(position_wise, end - start))
    return upper_bound_combinations / total_combinations


def limit_condition(p, bins, measurements):
    """
    Calculate the condition:

    :param p: a **sorted** partition
    :param bins: minimum number of bins that should be possible to fill (D+1 or K)
    :param measurements: minimum number of measurements per bin (K or D+1)
    :return:
        true if the condition is satisfied
    """
    extra = p[:-bins]
    missing = np.clip(measurements - np.array(p[-bins:]), a_min=0, a_max=None)
    return np.sum(extra) >= np.sum(missing)


def partitions(n, bins):
    return _partitions(n, bins, n)


def _partitions(n, bins, previous):
    if n == 0:
        yield (0, ) * (bins)
    elif bins * previous >= n:
        for i in range(max(0, n - previous), n):
            for p in _partitions(i, bins - 1, n - i):
                yield (n - i, ) + p


def partition_frequency(partition):
    total = special.factorial(len(partition))
    _, counts = np.unique(partition, return_counts=True)
    for c in counts:
        total = total / special.factorial(c)
    return total


def probability_upper_bound_any_measurements(n_dimensions,
                                             n_constraints,
                                             n_positions,
                                             n_anchors,
                                             n_measurements,
                                             position_wise=False):
    """Calculate upper bound on the probability of matrix being full rank,
    assuming that the number of measurements is exactly n_constraints * (n_dimensions + 1).
    This assumption allows for speed up calculations.

    :param n_dimensions: number of dimensions D
    :param n_constraints: number of constrains K
    :param n_positions: number of positions along trajectory N
    :param n_anchors: number of anchors M
    :param position_wise: if True, calculates the upper bound based on partitions of positions and not anchors.
    The equations are the same for both cases, but for readability purposes the naming convention for partition of
    anchors is used rather than abstract notation
    :param n_measurements: TODO
    :return:
        float: upper bound on probability of the left hand side of the matrix being full rank
    """

    if position_wise:
        (n_dimensions, n_constraints) = (n_constraints - 1, n_dimensions + 1)
        (n_anchors, n_positions) = (n_positions, n_anchors)

    start = time.time()
    upper_bound_combinations = 0
    for partition in partitions(n_measurements, n_anchors):
        if limit_condition(partition[::-1], n_dimensions + 1, n_constraints):
            new_combinations = partition_frequency(partition)
            for k in partition:
                new_combinations *= special.binom(n_positions, k)
            upper_bound_combinations += new_combinations

    total_combinations = special.binom(n_positions * n_anchors, n_measurements)
    end = time.time()
    if end - start > 1:
        print("Upper bound (any), position {}, elapsed time: {:.2f}s".format(position_wise, end - start))
    return upper_bound_combinations / total_combinations


# TODO from the plots this lower bound does not seem to be lower.
# It seems to calculate the upper bound minus a constant?
def probability_lower_bound(n_dimensions, n_constraints, n_positions, n_anchors):

    start = time.time()
    max_index = n_constraints + 1
    n_measurements = n_constraints * (n_dimensions + 1)
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


def left_independence_estimation(n_constraints, min_anchors, poisson_mean, repetitions=10000, use_limits=False):
    """
    Estimate joint and marginal probabilities of the left hand side matrix satisfying
    certain anchor and positions conditions necessary for the matrix to be full rank


    :param n_constraints: number of constrains K
    :param min_anchors: minimum number of anchors (D+1)
    :param poisson_mean: mean of the poisson variable added n_constraints
    and min_anchors to obtain number of positions and number of anchors
    :param repetitions: number of times to generate data
    :param use_limits: if True uses new limit conditions, otherwise use two out of four old conditions
    :return:
        tuple (number of good row configurations, number of good row and column configurations,
        number of entirely good row configurations, total number of tests made
    """

    anchors_ok = 0
    positions_ok = 0
    both_ok = 0
    total = 0
    for _ in range(repetitions):
        n_anchors = min_anchors + np.random.poisson(poisson_mean)
        n_positions = n_constraints + np.random.poisson(poisson_mean)
        p = min(1, (min_anchors * n_constraints) / (n_anchors * n_positions))
        matrix = np.random.binomial(1, p, size=(n_positions, n_anchors))
        # Proceed only if we have enough measurements
        if np.sum(matrix) >= n_constraints * min_anchors:
            total += 1
            positions_feasible = True
            anchors_feasible = True
            if use_limits:
                anchors_feasible = limit_condition(np.sort(np.sum(matrix, axis=0)), min_anchors, n_constraints)
                positions_feasible = limit_condition(np.sort(np.sum(matrix, axis=1)), n_constraints, min_anchors)
            else:
                for part_nr in range(2**n_positions):
                    partition = np.unravel_index(part_nr, [2] * n_positions)
                    if sum(partition) == (n_positions - (n_constraints - 1)):
                        # it sets feasible to false!
                        if np.sum(np.multiply(partition, np.sum(matrix, axis=1))) < min_anchors:
                            positions_feasible = False
                            break
                for part_nr in range(2**n_anchors):
                    partition = np.unravel_index(part_nr, [2] * n_anchors)
                    if sum(partition) == (n_anchors - (min_anchors - 1)):
                        # it sets feasible to false!
                        if np.sum(np.multiply(partition, np.sum(matrix, axis=0))) < n_constraints:
                            anchors_feasible = False
                            break

            if anchors_feasible:
                anchors_ok += 1
            if positions_feasible:
                positions_ok += 1
            if anchors_feasible and positions_feasible:
                both_ok += 1

    return anchors_ok, positions_ok, both_ok, total


def matrix_rank_experiment(params):

    n_measurements = 0
    n_positions = 0
    n_anchors_list = [max(params["n_dimensions"] * n, params["n_dimensions"] + 1) for n in params["n_anchors_list"]]
    params["n_anchors_list"] = n_anchors_list
    if "fixed_n_measurements" in params:
        second_list = list(range(params["min_positions"], params["max_positions"]))
        if params["fixed_n_measurements"] == 0:
            n_measurements = (params["n_dimensions"] + 1) * params["n_constraints"]
            if params["full_matrix"]:
                n_measurements += params["n_constraints"] - 1
            params["fixed_n_measurements"] = n_measurements
        else:
            n_measurements = params["fixed_n_measurements"]
        params["second_key"] = "m{}".format(n_measurements)
    else:
        second_list = list(
            range(params["n_dimensions"] * params["n_constraints"],
                  (params["n_dimensions"] + 1) * params["min_positions"] + 1))
        n_positions = params["n_positions"]
        params["second_key"] = "p{}".format(n_positions)
    params["second_list"] = second_list

    ranks = np.zeros((len(second_list), len(n_anchors_list), params["n_repetitions"]))
    anchor_condition = np.zeros_like(ranks)
    frame_condition = np.zeros_like(ranks)
    wrong_matrices = []
    for a_idx, n_anchors in enumerate(n_anchors_list):
        anchors = get_anchors(n_anchors, params["n_dimensions"])
        for second_idx, second_param in enumerate(second_list):
            if "fixed_n_measurements" in params:
                n_positions = second_param
            else:
                n_measurements = second_param
            for r in range(params["n_repetitions"]):
                frame = get_frame(params["n_constraints"], n_positions)
                try:
                    idx_a, idx_f = random_indexes(n_anchors, n_positions, n_measurements)
                    if params["full_matrix"]:
                        constraints = get_full_matrix(idx_a, idx_f, anchors, frame)
                    else:
                        constraints = get_left_submatrix(idx_a, idx_f, anchors, frame)
                    ranks[second_idx, a_idx, r] = np.linalg.matrix_rank(constraints)
                    measurement_matrix = indexes_to_matrix(idx_a, idx_f, n_anchors, n_positions)
                    if limit_condition(
                            np.sort(np.sum(measurement_matrix, axis=0)), params["n_constraints"],
                            params["n_dimensions"] + 1):
                        anchor_condition[second_idx, a_idx, r] = 1
                    if limit_condition(
                            np.sort(np.sum(measurement_matrix, axis=1)), params["n_dimensions"] + 1,
                            params["n_constraints"]):
                        frame_condition[second_idx, a_idx, r] = 1
                    if ranks[second_idx, a_idx, r] < params["n_constraints"] * (params["n_dimensions"] + 1):
                        if frame_condition[second_idx, a_idx, r] * anchor_condition[second_idx, a_idx, r] == 1:
                            wrong_matrices.append({
                                "constraints": constraints,
                                "measurements": measurement_matrix,
                                "second_idx": second_idx,
                                "a_idx": a_idx
                            })

                except ValueError as e:
                    ranks[second_idx, a_idx, r] = np.NaN
                    print(e)
                    break

    params["anchor_condition"] = anchor_condition
    params["frame_condition"] = frame_condition
    params["wrong_matrices"] = wrong_matrices

    return ranks, params


def plot_results(
        ranks,
        params,
        directory="results/ranks/",
        save=False,
):

    key = "_d{}_c{}_{}_full{}".format(params["n_dimensions"], params["n_constraints"], params["second_key"],
                                      params["full_matrix"])
    max_rank = (params["n_dimensions"] + 1) * params["n_constraints"]

    if params["full_matrix"]:
        max_rank += params["n_constraints"] - 1
    params["max_rank"] = max_rank
    n_repetitions = ranks.shape[2]
    x = np.array(params["second_list"])
    if "fixed_n_measurements" not in params:
        x = x / max_rank

    f, ax = plt.subplots()
    for a_idx, n_anchors in enumerate(params["n_anchors_list"]):
        plt.plot(
            x,
            np.mean(ranks[:, a_idx, :], axis=1) / max_rank,
            label="mean rank, {} anchors".format(n_anchors),
            color="C{}".format(a_idx),
            linestyle='dashed')
        plt.step(
            x,
            np.sum(ranks[:, a_idx, :] >= max_rank, axis=1) / n_repetitions,
            label="probability, {} anchors".format(n_anchors),
            color="C{}".format(a_idx),
            where='post')
    if "fixed_n_measurements" in params:
        plt.xlabel("number of positions")
    else:
        plt.xlabel("number of measurements")
        formatter_text = '%g (D+1)K + (K-1)' if params["full_matrix"] else '%g (D+1)K'
        ax.xaxis.set_major_formatter(tck.FormatStrFormatter(formatter_text))
        ax.xaxis.set_major_locator(tck.MultipleLocator(base=1))
    plt.grid()
    plt.legend()
    params["directory"] = directory
    if save:
        plt.ylim(bottom=0)
        matrix_type = "full" if params["full_matrix"] else "left"
        fname = directory + matrix_type + "_matrix_anchors" + key + ".pdf"
        make_dirs_safe(fname)
        plt.savefig(fname)
