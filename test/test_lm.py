#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_lm.py: Test LM optimization algorithm.
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
        sigma = 1e-10

        D_noisy = add_noise(self.D_topright, noise_sigma=sigma)

        C_k_vec = self.traj.coeffs.reshape((-1, ))
        jacobian = cost_jacobian(C_k_vec, D_noisy, self.anchors, self.basis)

        cost = cost_function(C_k_vec, D_noisy, self.anchors, self.basis)
        N = len(cost)
        Kd = len(C_k_vec)

        # make delta small enough but not too small.
        deltas = list(np.logspace(-15, -1, 10))[::-1]
        previous_jac = 1000
        convergence_lim = 1e-5

        for delta in deltas:
            jacobian_est = np.empty((N, Kd))
            for k in range(Kd):
                C_k_delta = C_k_vec.copy()
                C_k_delta[k] += delta
                cost_delta = cost_function(C_k_delta, D_noisy, self.anchors, self.basis)
                jacobian_est[:, k] = (cost_delta - cost) / delta

            new_jac = jacobian_est
            difference = np.sum(np.abs(previous_jac - new_jac))
            print('convergence:', difference)
            if np.sum(np.abs(new_jac)) < EPS:
                print('new jacobian is all zero! use previous jacobian.')
                break

            elif difference < convergence_lim:
                print(f'converged at {delta}.')
                previous_jac = new_jac
                break
            else:  # not converged yet.
                previous_jac = new_jac
        jacobian_est = previous_jac
        print('===== first element =====:')
        print('jacobian est vs. real:', jacobian_est[0, 0], jacobian[0, 0])
        print('difference', jacobian_est[0, 0] - jacobian[0, 0])
        print('==== total difference ===:')
        print(np.sum(np.abs(jacobian_est - jacobian)))


if __name__ == "__main__":
    unittest.main()
