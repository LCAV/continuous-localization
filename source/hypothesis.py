import numpy as np
from scipy import special
import time
import warnings
import measurements as m


def get_frame(n_constraints, n_positions):
    """Generate frame (basis) the fast way, without crating the trajectory.
    It speeds up simulations compared to Trajectory.get_basis.

    :param n_constraints: mode size
    :param n_positions: number of samples

    :return: a frame - an evaluated uniformly basis of bandlimited functions
    of size (n_constraints x n_positions)
    """
    Ks = np.arange(n_constraints).reshape((n_constraints, 1))
    Ns = np.arange(n_positions).reshape((n_positions, 1))
    return np.cos(Ks @ Ns.T * np.pi / n_positions)


def get_left_submatrix(idx_a, idx_f, anchors, frame):
    """Generated left submatrix in the extended form.

    The extended form means that anchors are extended by 1 making them n_dimensions+1 dimensional.
    Trows of the matrix are tensor product of extended anchors and frame vectors,
    which means that the whole matrix is (n_dimensions+1) * n_constraints x n_measurements dimensional.
    This is because it seems to be a more natural representation - for localising just one point,
    the full matrix is reduced to the (extended) left submatrix.

    :param idx_a: list of anchor indexes for each measurement
    :param idx_f: list of frame indexes for each measurement
    :param anchors: matrix of all available anchors, of size n_dimensions x n_anchors
    :param frame: matrix of all frame vectors, of size n_points x n_constraints

    :return: left part of the constrain matrix of size (n_dimensions+1) * n_constraints x n_measurements
    """

    f_vect = [frame[:, i] for i in idx_f]
    a_extended = [np.concatenate([anchors[:, a], [1]]) for a in idx_a]
    matrices = [(a[:, None] @ f[None, :]).flatten() for (f, a) in zip(f_vect, a_extended)]
    return np.array(matrices)


def get_reduced_right_submatrix(idx_f, frame):
    """Generated right submatrix in the reduced form.

    The extended form means the tensor products of frame vectors are reduced to a basis,
    which means that the size of the submatrix is (n_constraints - 1) x n_measurements.

    :param idx_f: list of frame indexes for each measurement
    :param frame: matrix of all frame vectors, of size n_constraints x n_positions

    :return: right part of the constrain matrix of size (n_constraints - 1) x n_measurements.
    """

    n_constraints, n_positions = frame.shape
    Ks = np.arange(n_constraints, 2 * n_constraints - 1).reshape((n_constraints - 1, 1))
    Ns = np.arange(n_positions).reshape((n_positions, 1))
    extended_frame = np.cos(Ks @ Ns.T * np.pi / n_positions)
    vectors = [extended_frame[:, idx] for idx in idx_f]
    matrix = np.array(vectors)
    return np.array(matrix)


def get_full_matrix(idx_a, idx_f, anchors, frame):
    """ Get full matrix in the reduced form (that can be invertible)

    :param idx_a: list of anchor indexes for each measurement
    :param idx_f: list of frame indexes for each measurement
    :param anchors: matrix of all available anchors, of size n_dimensions x n_anchors
    :param frame: matrix of all frame vectors, of size n_points x n_constraints

    :return: the constrain matrix ((n_dimensions+2) * n_constraints - 1 ) x n_measurements
    """
    return np.concatenate([get_left_submatrix(idx_a, idx_f, anchors, frame),
                           get_reduced_right_submatrix(idx_f, frame)],
                          axis=1)


def random_indexes(n_anchors, n_positions, n_measurements, one_per_time=False):
    if one_per_time:
        if n_positions < n_measurements:
            raise ValueError("to many measurements {}>{} requested".format(n_measurements, n_positions))
        idx_f = np.random.choice(n_positions, n_measurements, replace=False)
        idx_a = np.random.choice(n_anchors, n_measurements, replace=True)
    else:
        if n_positions * n_anchors < n_measurements:
            raise ValueError("to many measurements {}>{}x{} requested ".format(n_measurements, n_positions, n_anchors))
        indexes = np.random.choice(n_positions * n_anchors, n_measurements, replace=False)
        idx_a, idx_f = np.unravel_index(indexes, (n_anchors, n_positions))
    return idx_a.tolist(), idx_f.tolist()


def indexes_to_matrix(idx_a, idx_f, n_anchors, n_positions):
    matrix = np.zeros((n_anchors, n_positions))
    matrix[idx_a, idx_f] = 1
    return matrix


def limit_condition(p, bins, measurements):
    """
    Calculate the condition from Theorem 1.

    :param p: a partition **sorted in a descending order**
    :param bins: minimum number of bins that should be possible to fill (D+1 or K)
    :param measurements: minimum number of measurements per bin (K or D+1)
    :return:
        true if the condition is satisfied
    """
    missing = np.clip(measurements - np.array(p[:bins]), a_min=0, a_max=None)
    return np.sum(p[bins:]) >= np.sum(missing)


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
    return total / np.prod(special.factorial(counts))


def probability_upper_bound(n_dimensions,
                            n_constraints,
                            n_positions,
                            n_anchors,
                            n_measurements,
                            position_wise=False,
                            full_matrix=False):
    """Calculate upper bound on the probability of matrix being full rank,
    assuming that the number of measurements is exactly n_constraints * (n_dimensions + 1).
    This assumption allows for speed up calculations.

    Performance remarks: the anchor bound computation time depends heavily on the number of anchors,
    and similarly the position bound computation depends heavily on the number of positions/times.
    Using the limits for number of positions/times going to infinity does not seem to speed up the calculation,
    and in practice n_measurements 20x n_constrains is quite close to the infinity bound, and 1000x n_constrains
    is indistinguishable from infinity bound on the plot. Thus, the infinity bound is there mostly to test the theory.

    :param n_dimensions: number of dimensions D
    :param n_constraints: number of constrains K
    :param n_positions: number of positions along trajectory N, if infinity then the model when each measurement is
    taken at a different position/time is assumed.
    :param n_anchors: number of anchors M, if infinity then the model when each anchor is used only in one measurement
    is assumed. This is not realistic and added only to preserve symmetry (see below).
    :param position_wise: if True, calculates the upper bound based on partitions of positions and not anchors.
    The equations are the same for both cases, but for readability purposes the naming convention for partition of
    anchors is used rather than abstract notation
    :param n_measurements: total number of measurements taken
    :param full_matrix: if true, returns the  bound on probability of full matrix being of maximal rank. The two
    necessary conditions are the left submatrix being full rank and having at least (D+2)K-1 measurements
    :return:
        float: upper bound on probability of the left hand side of the matrix being full rank
    """

    # Because the bound is symmetric, we can swap constrains and dimensions
    # to obtain the second type of bound
    if position_wise:
        (n_dimensions, n_constraints) = (n_constraints - 1, n_dimensions + 1)
        (n_anchors, n_positions) = (n_positions, n_anchors)

    if np.isinf(n_anchors):
        return 1.0

    start = time.time()
    if full_matrix:
        if n_measurements < (n_dimensions + 2) * n_constraints - 1:
            return 0
    upper_bound_sum = 0
    for partition in partitions(n_measurements, n_anchors):
        if limit_condition(partition, n_dimensions + 1, n_constraints):
            new_combinations = partition_frequency(partition)
            # use limit formulation that does not use n_positions
            if np.isinf(n_positions):
                new_combinations /= np.prod(special.factorial(partition))
            # use general formulation
            else:
                new_combinations *= np.prod(special.binom(n_positions, partition))
            upper_bound_sum += new_combinations

    end = time.time()
    if end - start > 1:
        print("Upper bound, position {}, measurements {}, elapsed time: {:.2f}s".format(
            n_measurements, position_wise, end - start))

    if np.isinf(n_positions):
        common_factor = special.factorial(n_measurements) / (n_anchors**n_measurements)
    else:  # the common factor in this case is the total number of partitions
        common_factor = 1. / special.binom(n_positions * n_anchors, n_measurements)
    return upper_bound_sum * common_factor


def matrix_rank_experiment(params):
    """Run simulations to estimate probability of matrix being full rank for different number of measurements

     :param params: all parameters of the simulation, might contain:
        n_dimensions: number of dimensions
        n_constraints: number of constrains / degrees of freedom of the trajectory
        n_positions: number of positions at which measurements are taken
        n_repetitions: number of runs with the same parameters
        full_matrix: if True simulate the whole matrix, otherwise only the left part
        n_anchors_list: list of number number of anchors f
        one_per_time: if True, number of measurements per time/position is limited to 1
        fixed_n_measurements: if present, the number of measurements is fixed, and the number of positions/times very
        incompatible with `one_per_time`
     """

    n_measurements = 0
    n_positions = 0
    n_anchors_list = [max(params["n_dimensions"] * n, params["n_dimensions"] + 1) for n in params["n_anchors_list"]]
    params["n_anchors_list"] = n_anchors_list
    if 'one_per_time' not in params:
        params['one_per_time'] = False

    # prepare parameters for when the number of measurements is fixed
    if "fixed_n_measurements" in params:
        if params["one_per_time"]:
            warnings.warn("It does not make sense to fix number of measurements and have only one measurement per "
                          "time. All iterations will give the same result.")
        if params["fixed_n_measurements"] == 0:
            n_measurements = (params["n_dimensions"] + 1) * params["n_constraints"]
            if params["full_matrix"]:
                n_measurements += params["n_constraints"] - 1
            params["fixed_n_measurements"] = n_measurements
        else:
            n_measurements = params["fixed_n_measurements"]
        params["min_positions"] = n_measurements if params['one_per_time'] else int(
            np.ceil(n_measurements / np.min(n_anchors_list)))
        params["second_key"] = "m{}".format(n_measurements)
        second_list = list(range(params["min_positions"], params["max_positions"]))

    # prepare parameters for when the number of positions is fixed
    else:
        n_positions = params["n_positions"]
        max_measurements = (1 if params["one_per_time"] else (np.min(params["n_anchors_list"]))) * n_positions
        second_list = list(range(params["n_dimensions"] * params["n_constraints"], max_measurements))
        params["second_key"] = "p{}".format(n_positions)
    params["second_list"] = second_list

    ranks = np.zeros((len(second_list), len(n_anchors_list), params["n_repetitions"]))
    anchor_condition = np.zeros_like(ranks)
    frame_condition = np.zeros_like(ranks)
    wrong_matrices = []
    for a_idx, n_anchors in enumerate(n_anchors_list):
        anchors = m.create_anchors(n_anchors=n_anchors, dim=params["n_dimensions"], check=True)
        # iterate over whatever the second parameter is (number of positions or number of measurements)
        for second_idx, second_param in enumerate(second_list):
            if "fixed_n_measurements" in params:
                n_positions = second_param
            else:
                n_measurements = second_param
            for r in range(params["n_repetitions"]):
                frame = get_frame(params["n_constraints"], n_positions)
                try:
                    idx_a, idx_f = random_indexes(n_anchors,
                                                  n_positions,
                                                  n_measurements,
                                                  one_per_time=params["one_per_time"])
                    if params["full_matrix"]:
                        constraints = get_full_matrix(idx_a, idx_f, anchors, frame)
                    else:
                        constraints = get_left_submatrix(idx_a, idx_f, anchors, frame)
                    ranks[second_idx, a_idx, r] = np.linalg.matrix_rank(constraints)
                    measurement_matrix = indexes_to_matrix(idx_a, idx_f, n_anchors, n_positions)
                    if limit_condition(-np.sort(-np.sum(measurement_matrix, axis=0)), params["n_constraints"],
                                       params["n_dimensions"] + 1):
                        anchor_condition[second_idx, a_idx, r] = 1
                    if limit_condition(-np.sort(-np.sum(measurement_matrix, axis=1)), params["n_dimensions"] + 1,
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
    params["max_rank"] = (params["n_dimensions"] + 1) * params["n_constraints"]
    if params["full_matrix"]:
        params["max_rank"] += params["n_constraints"] - 1
    return ranks, params
