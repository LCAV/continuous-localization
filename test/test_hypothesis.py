#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import common

from collections import Counter
import unittest

from hypothesis import *


class TestGetAnchors(unittest.TestCase):
    def test_dimensions(self):
        n_anchors = 5
        self.assertEqual((2, n_anchors), get_anchors(n_anchors).shape)


class TestGetFrame(unittest.TestCase):
    def test_dimensions(self):
        n_constrains = 5
        n_positions = 13
        self.assertEqual((n_constrains, n_positions), get_frame(n_constrains, n_positions).shape)


class TestGetLeftSubmatrix(unittest.TestCase):
    def test_dimensions(self):
        n_anchors = 3
        n_constrains = 5
        n_positions = 13
        ind_a = [0] * 8
        ind_b = ind_a
        anchors = get_anchors(n_anchors)
        frame = get_frame(n_constrains, n_positions)
        self.assertEqual((len(ind_a), (anchors.shape[0] + 1) * n_constrains),
                         get_left_submatrix(ind_a, ind_b, anchors, frame).shape)


class TestGetRightSubmatrix(unittest.TestCase):
    def test_dimensions(self):
        n_constrains = 5
        n_positions = 13
        idx_f = [0] * 8
        frame = get_frame(n_constrains, n_positions)
        self.assertEqual((len(idx_f), n_constrains - 1), get_reduced_right_submatrix(idx_f, frame).shape)


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
        self.assertTrue(([4, 4, 3] == ranks[:, 0, 0]).all())
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


class TestBounds(unittest.TestCase):
    def test_few_anchors(self):
        n_dimensions = 2
        n_constrains = 5
        for n_positions in [15, 20, 25]:
            exact = probability_few_anchors(n_dimensions, n_constrains, n_positions)
            upper = probability_upper_bound(n_dimensions, n_constrains, n_positions, n_dimensions + 1)
            self.assertEqual(exact, upper)

    def test_few_anchors_any_measurements(self):
        n_dimensions = 2
        n_constrains = 5
        n_positions = 15
        exact = probability_few_anchors(n_dimensions, n_constrains, n_positions)
        upper = probability_upper_bound_any_measurements(
            n_dimensions, n_constrains, n_positions, n_dimensions + 1, n_measurements=(n_dimensions + 1) * n_constrains)
        self.assertEqual(exact, upper)

    def test_many_anchors(self):
        n_dimensions = 2
        n_constrains = 5
        n_positions = 20
        n_anchors = 5
        inefficient = probability_upper_bound(n_dimensions, n_constrains, n_positions, n_anchors)
        efficient = probability_upper_bound_any_measurements(
            n_dimensions, n_constrains, n_positions, n_anchors, n_measurements=(n_dimensions + 1) * n_constrains)
        self.assertEqual(efficient, inefficient)

    def test_limit_condition(self):
        part = (5, 5, 4, 1, 0)
        self.assertTrue(limit_condition(part[::-1], 3, 4))
        self.assertTrue(limit_condition(part[::-1], 3, 5))
        self.assertFalse(limit_condition(part[::-1], 4, 2))


if __name__ == '__main__':
    unittest.main()
