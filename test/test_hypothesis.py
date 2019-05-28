#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from common import test_prepare
test_prepare()

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


if __name__ == '__main__':
    unittest.main()
