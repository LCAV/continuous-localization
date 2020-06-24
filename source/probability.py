# -*- coding: utf-8 -*-
"""
probability.py: Functions to estimate the probability [T_A, T_B] being maximal rank,
using upper bounds and simulations.

"""

import time
import warnings

import numpy as np
from scipy import special

import constraints as c
import measurements as m


def random_indexes(n_anchors, n_times, n_measurements, one_per_time=False):
    """ Generate a random subset of n_measurements, uniformly over all subsets of this size.

    :param n_anchors: number of anchors available
    :param n_times: number of positions/time samples available
    :param n_measurements: total number of measurements to generate
    :param one_per_time: bool, if true return at most one measurement per position/time sample
    :return: a pair of lists (anchor indexes, frame indexes), both of lenght n_measurements
    """
    if one_per_time:
        if n_times < n_measurements:
            raise ValueError("to many measurements {}>{} requested".format(n_measurements, n_times))
        idx_f = np.random.choice(n_times, n_measurements, replace=False)
        idx_a = np.random.choice(n_anchors, n_measurements, replace=True)
    else:
        if n_times * n_anchors < n_measurements:
            raise ValueError("to many measurements {}>{}x{} requested ".format(n_measurements, n_times, n_anchors))
        indexes = np.random.choice(n_times * n_anchors, n_measurements, replace=False)
        idx_a, idx_f = np.unravel_index(indexes, (n_anchors, n_times))
    return idx_a.tolist(), idx_f.tolist()


def indexes_to_matrix(idx_a, idx_f, n_anchors, n_times):
    matrix = np.zeros((n_anchors, n_times))
    matrix[idx_a, idx_f] = 1
    return matrix


def full_rank_condition(p, bins, measurements):
    """
    Calculate the condition from Theorem 1 in Relax and Recover paper.

    :param p: a partition (does not need to be sorted anymore)
    :param bins: minimum number of bins that should be possible to fill (D+1 or K)
    :param measurements: minimum number of measurements per bin (K or D+1)

    :return: true if the condition is satisfied
    """

    total = np.sum(np.clip(p, a_min=None, a_max=measurements))
    return total >= bins * measurements


def partitions(n, n_bins):
    """ Generator of partitions of number n into n_bins bins

    Generates partitions in inverse-lexicographical order. Does not generate duplicates, that is among all
    partitions that are permutations of a partition p, it generates only one, the first in inverse-lexicographical
    order, for example if n=5 and n_bins=3, it will generate (5, 0, 0), but not (0, 5, 0), nor (0, 0, 5).

    This behaviour improves speed of simulations that do not depend on the order of elements in the partition (for
    example, order in which anchors are numbered does not matter)

    :param n: number to divide into permutations (number of balls to distribute)
    :param n_bins: number of bins to divide into (number of urns)
    :return: a n_bins-tuple, next partition in the inverse-lexicographical order.
    """

    return _partitions(n, n_bins, n)


def _partitions(n, n_bins, previous):
    """Helper function for partitions"""

    if n == 0:
        yield (0, ) * (n_bins)
    elif n_bins * previous >= n:
        for i in range(max(0, n - previous), n):
            for p in _partitions(i, n_bins - 1, n - i):
                yield (n - i, ) + p


def partition_frequency(partition):
    """ Calculate the number of partitions that are permutations of a given partition.

    If all entries of a length n partition are different, then there are n! permutations.
    But if some entries repeat, we would end up counting some permutations twice, so we divide
    by the number of permutations within each repeating group

    :param partition: a m-tuple, a partition
    :return: float, number of partitions that are permutations of partition.
    """
    total = special.factorial(len(partition))
    _, counts = np.unique(partition, return_counts=True)
    return total / np.prod(special.factorial(counts))


def probability_upper_bound(n_dimensions,
                            n_constraints,
                            n_times,
                            n_anchors,
                            n_measurements,
                            position_wise=False,
                            full_matrix=False):
    """Calculate upper bound on the probability of matrix being full rank.

    Can be used to calculate upper bounds anchor-wise or position-wise, since the problem is symmetric.
    In the first case, it iterates over partitions of n_measurements into n_anchors bins, and for each
    partition checks if `full_rank_condition` (eq (8), Theorem 1, Relax and Recover paper) is satisfied.
    If yes, then increase the count of "passing" partitions by the number of permutations of this partition.
    Finally obtain probability by dividing the "passing" partitions by the total number of partitions.

    Performance remarks: the anchor bound computation time depends heavily on the number of anchors,
    and similarly the position bound computation depends heavily on the number of positions/times.
    Using the limits for number of positions/times going to infinity does not seem to speed up the calculation,
    and in practice n_measurements 20x n_constrains is quite close to the infinity bound, and 1000x n_constrains
    is indistinguishable from infinity bound on the plot. Thus, the infinity bound is there mostly to test the theory.

    :param n_dimensions: number of dimensions D
    :param n_constraints: number of constrains K
    :param n_times: number of positions along trajectory N, if infinity then the model when each measurement is
        taken at a different position/time is assumed.
    :param n_anchors: number of anchors M, if infinity then the model when each anchor is used only in one measurement
        is assumed. This is not realistic and added only to preserve symmetry (see below).
    :param position_wise: if True, calculates the upper bound based on partitions of positions and not anchors.
        The equations are the same for both cases, but for readability purposes the naming convention for partition of
        anchors is used rather than abstract notation
    :param n_measurements: total number of measurements taken
    :param full_matrix: if true, returns the  bound on probability of full matrix being of maximal rank. The two
        necessary conditions are the left submatrix being full rank and having at least (D+2)K-1 measurements

    :return: float, an upper bound on probability of the left hand side of the matrix being full rank
    """

    # Because the bound is symmetric, we can swap constrains and dimensions
    # to obtain the second type of bound
    if position_wise:
        (n_dimensions, n_constraints) = (n_constraints - 1, n_dimensions + 1)
        (n_anchors, n_times) = (n_times, n_anchors)

    # Check corner cases
    if np.isinf(n_anchors):  # (just for a speed up)
        return 1.0

    # condition (7) from Relax and Recover
    if full_matrix and (n_measurements < (n_dimensions + 2) * n_constraints - 1):
        return 0

    start = time.time()  # Time the main loop
    upper_bound_sum = 0
    for partition in partitions(n_measurements, n_anchors):
        if full_rank_condition(partition, n_dimensions + 1, n_constraints):
            new_combinations = partition_frequency(partition)
            if np.isinf(n_times):  # use limit formulation that does not use n_times
                new_combinations /= np.prod(special.factorial(partition))
            else:  # use general formulation
                new_combinations *= np.prod(special.binom(n_times, partition))
            upper_bound_sum += new_combinations

    end = time.time()
    if end - start > 1:
        print("Upper bound, position {}, measurements {}, elapsed time: {:.2f}s".format(
            n_measurements, position_wise, end - start))

    if np.isinf(n_times):
        n_anchors = int(n_anchors)  # floats and numpy its have to small range
        common_factor = special.factorial(n_measurements) / (n_anchors**n_measurements)
    else:  # the common factor in this case is the total number of partitions
        common_factor = 1. / special.binom(n_times * n_anchors, n_measurements)
    return upper_bound_sum * common_factor


def matrix_rank_experiment(n_dimensions,
                           n_constraints,
                           n_times,
                           n_repetitions,
                           full_matrix,
                           n_anchors_list,
                           one_per_time=False):
    """Run simulations to estimate probability of matrix being full rank for different number of measurements


     :param n_dimensions: number of dimensions
     :param n_constraints: number of constrains / degrees of freedom of the trajectory
     :param n_times: number of positions at which measurements are taken
     :param n_repetitions: number of runs with the same parameters
     :param full_matrix: if True simulate the whole matrix, otherwise only the left part
     :param n_anchors_list: list of numbers of anchors to iterate over
     :param  one_per_time: if True, number of measurements per time/position is limited to 1
     """

    params = locals()

    n_anchors_list = np.array(n_anchors_list)
    n_anchors_list = n_anchors_list[n_anchors_list > n_dimensions]
    if len(n_anchors_list) == 0:
        raise ValueError('Number of anchors must be bigger than number of dimensions')
    if len(n_anchors_list) < len(params["n_anchors_list"]):
        print("Some numbers of anchors are smaller than number of dimensions +1: cropping anchor list")
        params["n_anchors_list"] = n_anchors_list

    max_measurements = (1 if one_per_time else (np.min(n_anchors_list))) * n_times + 1
    params["n_measurements_list"] = list(range(n_dimensions * n_constraints, max_measurements))

    ranks = np.zeros((len(params["n_measurements_list"]), len(n_anchors_list), n_repetitions))
    for a_idx, n_anchors in enumerate(n_anchors_list):
        anchors = m.create_anchors(n_anchors=n_anchors, dim=n_dimensions, check=True)
        for m_idx, n_measurements in enumerate(params["n_measurements_list"]):
            for r in range(n_repetitions):
                frame = c.get_frame(n_constraints, n_times)
                try:
                    idx_a, idx_f = random_indexes(n_anchors, n_times, n_measurements, one_per_time=one_per_time)
                    if full_matrix:
                        constraints = c.get_reduced_constraints(idx_a, idx_f, anchors, frame)
                    else:
                        constraints = c.get_left_submatrix(idx_a, idx_f, anchors, frame, extended=True)
                    ranks[m_idx, a_idx, r] = np.linalg.matrix_rank(constraints)

                except ValueError as e:
                    ranks[m_idx, a_idx, r] = np.NaN
                    print(e)
                    break

    params["max_rank"] = (n_dimensions + 1) * n_constraints
    if full_matrix:
        params["max_rank"] += n_constraints - 1
    return ranks, params
