#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import common

import numpy as np
import unittest

from iterative_algorithms import build_up_algorithm
from iterative_algorithms import averaging_algorithm
from measurements import get_measurements
from trajectory import Trajectory


class TestIterative(unittest.TestCase):
    def setUp(self):
        self.t1 = Trajectory(dim=2, n_complexity=2, model='polynomial', coeffs=np.array([[0., 0.], [0., 1.]]))
        self.t2 = Trajectory(dim=2, n_complexity=2, model='polynomial', coeffs=np.array([[-3., 3.], [1., 0.]]))
        times1 = np.linspace(0, 1, 10)
        times2 = np.linspace(1, 2, 20)

        np.random.seed(1)
        self.anchors = 4 * np.random.rand(2, 5)

        b1, D1 = get_measurements(self.t1, self.anchors, seed=1, times=times1)
        b2, D2 = get_measurements(self.t2, self.anchors, seed=1, times=times2)
        self.F = np.hstack((b1, b2))
        self.D = np.vstack((D1, D2))
        self.times = np.r_[times1, times2]

    def test_averaging_algorithm(self):
        C_list, t_list = averaging_algorithm(self.D, self.anchors, self.F, self.times, t_window=1.0)

        self.assertTrue(np.allclose(C_list[0], self.t1.coeffs))
        self.assertTrue(np.allclose(C_list[-2], self.t2.coeffs))
        self.assertTrue(np.allclose(C_list[-1], self.t2.coeffs))

    def test_averaging_algorithm(self):
        C_list, t_list = build_up_algorithm(self.D, self.anchors, self.F, self.times, eps=1e-3)

        self.assertTrue(np.allclose(C_list[0], self.t1.coeffs))
        self.assertTrue(np.allclose(C_list[1], self.t2.coeffs))


if __name__ == '__main__':
    unittest.main()
