#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_lm.py: 
"""

import common

import numpy as np
import unittest

from baseline_solvers import *
from trajectory import Trajectory
from measurements import get_measurements, create_anchors, add_noise
from other_algorithms import cost_function, cost_jacobian

EPS = 1e-15


class TestLM(unittest.TestCase):
    def setUp(self):
        self.traj = Trajectory(n_complexity=5, dim=2)
        self.n_anchors = 4
        self.basis = []
        self.D_topright = []

    def set_measurements(self, seed=None):
        #  random trajectory and anchors
        self.traj.set_coeffs(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.anchors = create_anchors(self.traj.dim, self.n_anchors)

        # get measurements
        self.basis, self.D_topright = get_measurements(self.traj, self.anchors, seed=seed, n_samples=20)

    def test_cost_function(self):
        for i in range(100):
            self.set_measurements(seed=i)

            C_k_vec = self.traj.coeffs.reshape((-1, ))
            cost = cost_function(C_k_vec, self.D_topright, self.anchors, self.basis)
            self.assertTrue(np.all(cost < EPS))

    def test_cost_jacobian(self):
        i = 1
        self.set_measurements(seed=i)
        # make sigma very small to test if the cost function
        # behaves well at least around the optimum.
        sigma = 1e-1

        #
        delta = 1e-10  #1e-10
        D_noisy = add_noise(self.D_topright, noise_sigma=sigma)

        C_k_vec = self.traj.coeffs.reshape((-1, ))
        jacobian = cost_jacobian(C_k_vec, D_noisy, self.anchors, self.basis)
        print('jacobian')
        print(jacobian)

        cost = cost_function(C_k_vec, D_noisy, self.anchors, self.basis)
        N = len(cost)
        Kd = len(C_k_vec)
        for k in range(Kd):
            C_k_delta = C_k_vec.copy()
            C_k_delta[k] += delta
            cost_delta = cost_function(C_k_delta, D_noisy, self.anchors, self.basis)
            for n in range(N):
                print('two costs', cost_delta[n], cost[n])
                print('diff', cost_delta[n] - cost[n])
                jacobian_est = (cost_delta[n] - cost[n]) / delta
                jacobian_k = jacobian[n, k]
                print('jac est', jacobian_est)
                print('jac k', jacobian_k)
                break
            break


if __name__ == "__main__":
    unittest.main()
