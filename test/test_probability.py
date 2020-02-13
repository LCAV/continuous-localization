#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing functions used in hypothesis testing. The code below checks if the general
method is compatible with methods designed for special cases, that are no longer
relevant except for testing, see TestBounds class.
"""

import common

from collections import Counter
import unittest

from probability import *


class TestRandomIndexes(unittest.TestCase):
    def test_type(self):
        idx_a, idx_f = random_indexes(3, 2, n_measurements=1)
        self.assertTrue(isinstance(idx_a, list))
        self.assertTrue(isinstance(idx_f, list))
        self.assertEqual(1, len(idx_a))
        self.assertEqual(1, len(idx_f))

    def test_one_per_time(self):
        idx_a, idx_f = random_indexes(3, 5, n_measurements=4, one_per_time=True)
        self.assertTrue(all(count < 2 for count in Counter(idx_f).values()))
        self.assertTrue(all(count < 3 for count in Counter(idx_a).keys()))

    def test_many_per_time(self):
        np.random.seed(0)
        idx_a, idx_f = random_indexes(3, 5, n_measurements=4)
        self.assertEqual(2, Counter(idx_f)[1])


class TestMatrixRankExperiments(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(0)

    def test_left_single_run(self):
        experiment_params = {
            "n_dimensions": 1,
            "n_constraints": 2,
            "fixed_n_measurements": 0,
            "max_positions": 5,
            "n_repetitions": 1,
            "full_matrix": False,
            "n_anchors_list": [1],
        }
        ranks, params = matrix_rank_experiment(experiment_params)
        self.assertEqual((3, 1, 1), ranks.shape)
        # there might be seed changes that we don't want to care about,
        # but we want the ranks to be reasonable
        self.assertTrue(([2, 2, 2] <= ranks[:, 0, 0]).all())
        self.assertTrue(([4, 4, 4] >= ranks[:, 0, 0]).all())
        self.assertEqual(2, params["n_anchors_list"][0])
        self.assertEqual(4, params["fixed_n_measurements"])

    def test_full_single_run(self):
        experiment_params = {
            "n_dimensions": 1,
            "n_constraints": 2,
            "fixed_n_measurements": 0,
            "max_positions": 4,
            "n_repetitions": 1,
            "full_matrix": True,
            "n_anchors_list": [1],
        }
        ranks, params = matrix_rank_experiment(experiment_params)
        self.assertEqual((1, 1, 1), ranks.shape)
        self.assertEqual(5, ranks[0, 0, 0])
        self.assertEqual(5, params["fixed_n_measurements"])


class TestPartitions(unittest.TestCase):
    def test_sorted(self):
        for part in partitions(10, 4):
            self.assertTrue(list(part[::-1]) == sorted(part))

    def test_frequency(self):
        n = 10
        k = 4
        total = sum([partition_frequency(part) for part in partitions(n, k)])
        self.assertEqual(special.binom(n + k - 1, k - 1), total)


def probability_few_anchors_limit(n_dimensions, n_constraints, anchors_limit=False):
    """Calculate analytical limit for n_positions->infinity of probability_few_anchors.

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


def probability_min_anchors(n_dimensions, n_constraints, n_positions):
    """Calculate probability of size the smallest matrix being invertible with minimal number of anchors.

    This is equal to probability_upper_bound_min_measurements for n_anchors = n_dimensions + 1.

    Smallest matrix that can invertible has to have (n_dimensions+1)n_constraints rows,
    and the smallest number of needed anchors is n_dimensions + 1. The behaviour is qualitatively different
    for n_dimensions + 1 anchors than for more anchors."""

    total = special.binom((n_dimensions + 1) * n_positions, (n_dimensions + 1) * n_constraints)
    full = special.binom(n_positions, n_constraints)**(n_dimensions + 1)
    return full / total


def probability_upper_bound_min_measurements(n_dimensions, n_constraints, n_positions, n_anchors, position_wise=False):
    """Calculate upper bound on the probability of matrix being full rank,
    assuming that the number of measurements is exactly n_constraints * (n_dimensions + 1).
    This assumption allows for speed up calculations.

    This is equal to probability_upper_bound for n_measurements =  n_constraints * (n_dimensions + 1)

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
        print("Upper bound, position {}, positions {}, elapsed time: {:.2f}s".format(
            n_positions, position_wise, end - start))
    return upper_bound_combinations / total_combinations


class TestBounds(unittest.TestCase):
    """ Tests compatibility between probability_upper_bound and all the old bounds """
    def setUp(self) -> None:
        self.n_dimensions = 2
        self.n_constrains = 5
        self.n_positions = 20

    def test_min_anchors(self):
        """Test if the probability estimate for min number of anchors matches the expected limit"""
        limit = probability_few_anchors_limit(self.n_dimensions, self.n_constrains)
        many_positions = probability_min_anchors(self.n_dimensions, self.n_constrains, n_positions=120)
        self.assertAlmostEqual(limit, many_positions, places=3)

    def test_upper_bound(self):
        """Test if fast upper bound for n_dimensions + 1 anchors matches the exact probability"""
        for n_positions in [15, 20, 25]:
            exact = probability_min_anchors(self.n_dimensions, self.n_constrains, n_positions)
            upper = probability_upper_bound_min_measurements(self.n_dimensions, self.n_constrains, n_positions,
                                                             self.n_dimensions + 1)
            self.assertEqual(exact, upper)

    def test_upper_bound_general(self):
        """Test if general upper bound for n_dimensions + 1 anchors and minimum number of measurements matches the
        exact probability"""
        exact = probability_min_anchors(self.n_dimensions, self.n_constrains, self.n_positions)
        upper = probability_upper_bound(self.n_dimensions,
                                        self.n_constrains,
                                        self.n_positions,
                                        self.n_dimensions + 1,
                                        n_measurements=(self.n_dimensions + 1) * self.n_constrains)
        self.assertEqual(exact, upper)

    def test_many_anchors(self):
        """Test if fast and general upper bounds match for minimum number of measurements (for which the fast bound
        is defined"""
        for n_anchors in range(3, 6):
            inefficient = probability_upper_bound_min_measurements(self.n_dimensions, self.n_constrains,
                                                                   self.n_positions, n_anchors)
            efficient = probability_upper_bound(self.n_dimensions,
                                                self.n_constrains,
                                                self.n_positions,
                                                n_anchors,
                                                n_measurements=(self.n_dimensions + 1) * self.n_constrains)
            self.assertEqual(efficient, inefficient)

    def test_limit_condition(self):
        part = (5, 5, 4, 1, 0)
        self.assertTrue(full_rank_condition(part, 3, 4))
        self.assertTrue(full_rank_condition(part, 3, 5))
        self.assertFalse(full_rank_condition(part, 4, 2))

    def test_infinity_anchors(self):
        infinity = probability_upper_bound(self.n_dimensions,
                                           self.n_constrains,
                                           n_positions=30,
                                           n_anchors=np.Infinity,
                                           n_measurements=30)
        self.assertAlmostEqual(1, infinity)

    def test_infinity_positions(self):
        for n_anchors in range(3, 6):
            infinity = probability_upper_bound(self.n_dimensions,
                                               self.n_constrains,
                                               n_positions=100000000,
                                               n_anchors=n_anchors,
                                               n_measurements=15)

            large = probability_upper_bound(self.n_dimensions,
                                            self.n_constrains,
                                            n_positions=np.Infinity,
                                            n_anchors=n_anchors,
                                            n_measurements=15)
            self.assertAlmostEqual(large, infinity)


if __name__ == '__main__':
    unittest.main()
