#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import common

import unittest
from hypothesis import *
from scipy import special


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
        self.assertEqual(((anchors.shape[0] + 1) * n_constrains, len(ind_a)),
                         get_left_submatrix(ind_a, ind_b, anchors, frame).shape)


class TestGetRightSubmatrix(unittest.TestCase):
    def test_dimensions(self):
        n_constrains = 5
        n_positions = 13
        idx_f = [0] * 8
        frame = get_frame(n_constrains, n_positions)
        self.assertEqual((n_constrains - 1, len(idx_f)), get_reduced_right_submatrix(idx_f, frame).shape)


class TestRandomIndexes(unittest.TestCase):
    def test_type(self):
        a, b = random_indexes(3, 2, 1)
        self.assertTrue(isinstance(a, list))
        self.assertTrue(isinstance(b, list))


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
