#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import common

import numpy as np
import unittest

from measurements import get_measurements, create_mask
from other_algorithms import least_squares_lm, cost_function, calculate_error
from other_algorithms import pointwise_srls, get_grid, pointwise_rls
from solvers import trajectory_recovery
from trajectory import Trajectory

eps = 1e-10


class TestOthers(unittest.TestCase):
    def setUp(self):
        np.random.seed(2)
        n_complexity = 5
        self.traj = Trajectory(model='bandlimited')
        self.traj.set_n_complexity(n_complexity)
        self.traj.set_coeffs(1)

        self.anchors = np.random.rand(self.traj.dim, 10) * 10  # between 0 and 10.

        self.times = self.traj.get_times(n_samples=200)
        self.basis, self.D_gt = get_measurements(self.traj, self.anchors[:2, :], times=self.times)

        points_gt = self.traj.get_sampling_points(self.times)
        self.indices = range(len(self.times))[::self.traj.dim + 1]
        self.points_sub = points_gt[:, self.indices]

    def test_least_squares_lm(self):
        mask = create_mask(*self.D_gt.shape, strategy='single_time')
        D_sparse = self.D_gt * mask

        C_gt_vec = self.traj.coeffs.reshape((-1, ))
        cost_gt = cost_function(C_gt_vec, D_sparse, self.anchors[:2, :], self.basis)
        self.assertTrue(np.sum(np.abs(cost_gt)) < eps)

        Chat = self.traj.coeffs
        x0 = Chat.copy().reshape((-1, ))
        Cref = least_squares_lm(D_sparse, self.anchors, self.basis, x0)
        self.assertLess(calculate_error(Cref, self.traj.coeffs), eps)

    def test_pointwise_srls(self):
        points, __ = pointwise_srls(self.D_gt, self.anchors, self.traj, self.indices)
        points = np.array(points).T
        self.assertTrue(np.allclose(self.points_sub, points))

    def test_pointwise_rls(self):
        grid_size = 1.0
        grid = get_grid(self.points_sub, grid_size)

        points, __ = pointwise_rls(self.D_gt, self.anchors, self.traj, self.indices, grid=grid)
        points = points.T
        np.testing.assert_allclose(points, self.points_sub, atol=grid_size, rtol=grid_size)

    def test_single_rls_srls(self):
        from other_algorithms import RLS
        from pylocus.lateration import SRLS

        anchors = np.random.uniform(0, 10, size=(2, 4))
        grid_size = 0.1
        grid = get_grid(anchors, grid_size=grid_size)
        chosen_idx = np.random.choice(range(grid.shape[0]))
        pos_real = grid[chosen_idx, :]
        r2_real = np.linalg.norm(anchors - pos_real[:, None], axis=0)**2

        np.testing.assert_allclose(pos_real, RLS(anchors.T, r2_real, grid))
        r2_real = r2_real.reshape((-1, 1))
        weights = np.ones(r2_real.shape)
        np.testing.assert_allclose(pos_real, SRLS(anchors.T, weights, r2_real))


if __name__ == "__main__":
    unittest.main()
