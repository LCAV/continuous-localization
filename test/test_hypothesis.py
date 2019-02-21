import sys
sys.path.append('../')

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


class TestGetFullConstrains(unittest.TestCase):
    def test_dimensions(self):
        n_anchors = 3
        n_constrains = 5
        n_positions = 13
        ind_a = [0] * 8
        ind_b = ind_a
        anchors = get_anchors(n_anchors)
        frame = get_frame(n_constrains, n_positions)
        self.assertEqual(((2 + n_constrains)**2, len(ind_a)),
                         get_full_constrains(ind_a, ind_b, anchors, frame).shape)


class TestGetKDConstrains(unittest.TestCase):
    def test_dimensions(self):
        n_anchors = 3
        n_constrains = 5
        n_positions = 13
        ind_a = [0] * 8
        ind_b = ind_a
        anchors = get_anchors(n_anchors)
        frame = get_frame(n_constrains, n_positions)
        self.assertEqual(((2 * n_constrains), len(ind_a)),
                         get_upper_right_constrains(ind_a, ind_b, anchors, frame).shape)


class TestRandomIndexes(unittest.TestCase):
    def test_type(self):
        a, b = random_indexes(3, 2, 1)
        self.assertTrue(isinstance(a, list))
        self.assertTrue(isinstance(b, list))


if __name__ == '__main__':
    unittest.main()
