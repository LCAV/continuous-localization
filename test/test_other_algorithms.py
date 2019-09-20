#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import common

import numpy as np
import unittest

from measurements import get_measurements, create_mask
from other_algorithms import least_squares_lm, cost_function, calculate_error
from other_algorithms import pointwise_srls
from solvers import alternativePseudoInverse
from trajectory import Trajectory

eps = 1e-10


class TestOthers(unittest.TestCase):
    def setUp(self):
        np.random.seed(2)
        n_complexity = 5
        self.traj = Trajectory(model='bandlimited')
        self.traj.set_n_complexity(n_complexity)
        self.traj.set_coeffs(1)

        self.anchors = np.random.rand(self.traj.dim, 10) * 10

        self.times = self.traj.get_times(n_samples=200)
        self.basis, self.D_gt = get_measurements(self.traj, self.anchors[:2, :], times=self.times)

    def test_least_squares_lm(self):
        mask = create_mask(*self.D_gt.shape, strategy='single_time')
        D_sparse = self.D_gt * mask

        C_gt_vec = self.traj.coeffs.reshape((-1, ))
        cost_gt = cost_function(C_gt_vec, D_sparse, self.anchors[:2, :], self.basis)
        self.assertTrue(np.sum(np.abs(cost_gt)) < eps)

        # TODO why does this not work?
        # Chat = alternativePseudoInverse(D_sparse, self.anchors[:2, :], self.basis, weighted=True)
        #self.assertLess(calculate_error(Chat, self.traj.coeffs), eps)

        Chat = self.traj.coeffs
        x0 = Chat.copy().reshape((-1, ))
        Cref = least_squares_lm(D_sparse, self.anchors, self.basis, x0)
        self.assertLess(calculate_error(Cref, self.traj.coeffs), eps)

        #x0 = np.random.rand(*Chat.shape).reshape((-1, )) * 10
        #Crand = least_squares_lm(D_sparse, self.anchors, self.basis, x0)
        #self.assertLess(calculate_error(Crand, self.traj.coeffs), eps)

    def test_pointwise_srls(self):
        indices = range(len(self.times))
        points_gt = self.traj.get_sampling_points(self.times)
        points_sub = points_gt[:, ::self.traj.dim + 1]
        points = pointwise_srls(self.D_gt, self.anchors, self.basis, self.traj, indices)
        points = np.array(points).T
        self.assertTrue(np.allclose(points_sub, points))


if __name__ == "__main__":
    unittest.main()
