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
from other_algorithms import cost_function, cost_jacobian, least_squares_lm

EPS = 1e-15
N_IT = 100
VERBOSE = False


def test_around_local_minimum(x0, *args):
    delta = 1e-10
    res_minimum = cost_function(x0, *args)
    cost_minimum = 0.5 * np.sum(res_minimum**2)
    for i in range(len(x0)):
        x0_delta = x0.copy()
        x0_delta[i] += delta
        res_delta = cost_function(x0_delta, *args)
        cost_delta = 0.5 * np.sum(res_delta**2)
        difference = cost_delta - cost_minimum
        if difference < 0:
            assert abs(difference) < delta, f'difference negative and too big: {difference:.2e} > {EPS}'


class TestLM(unittest.TestCase):
    def setUp(self):
        self.traj = Trajectory(n_complexity=5, dim=2)
        self.n_anchors = 4
        self.basis = []
        self.D_topright = []

    def set_measurements(self, seed=None):
        self.traj.set_coeffs(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.anchors = create_anchors(self.traj.dim, self.n_anchors)

        # get measurements
        self.basis, self.D_topright = get_measurements(self.traj, self.anchors, seed=seed, n_samples=20)

    def test_cost_function(self):
        """ Test that cost for noiseless is indeed zero. """
        for i in range(N_IT):
            self.set_measurements(seed=i)

            C_k_vec = self.traj.coeffs.reshape((-1, ))
            cost = cost_function(C_k_vec, self.D_topright, self.anchors, self.basis)
            self.assertTrue(np.all(cost < EPS))

            test_around_local_minimum(C_k_vec, self.D_topright, self.anchors, self.basis)

    def test_convergence(self):
        """ Test that we converge correctly. """
        sigma = 0.01
        for i in range(N_IT):
            self.set_measurements(seed=i)

            D_noisy = add_noise(self.D_topright, noise_sigma=sigma)
            x0 = self.traj.coeffs.reshape((-1, ))
            cost0 = cost_function(x0, D_noisy, self.anchors, self.basis)

            xhat = least_squares_lm(D_noisy, self.anchors, self.basis, x0=x0, verbose=VERBOSE)
            xhat = xhat.reshape((-1, ))
            costhat = cost_function(xhat, D_noisy, self.anchors, self.basis)
            self.assertLessEqual(np.sum(costhat**2), np.sum(cost0**2))

            try:
                test_around_local_minimum(xhat, D_noisy, self.anchors, self.basis)
            except Exception as e:
                print(f'test_convergence failed at seed {i}')
                print('Error message:', e)

    def test_cost_jacobian(self):
        """ Test with finite differences that Jacobian is correct."""
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
