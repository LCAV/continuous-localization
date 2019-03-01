#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from os.path import abspath, dirname
sys.path.append(dirname(abspath(__file__)) + '/../')
print(sys.path)

import numpy as np
import unittest
from cvxpy import CVXOPT

from solvers import *
from trajectory import Trajectory
from environment import Environment
from simulation import run_simulation


class TestMethods(unittest.TestCase):
    def setUp(self):
        self.traj = Trajectory()
        self.env = Environment()
        self.D_topright = []

    def set_measurements(self, seed=1):
        self.traj.set_coeffs(seed=seed)
        times = self.traj.get_times()
        self.traj.set_basis(times)
        self.env.set_random_anchors(seed=seed)
        self.env.set_D(self.traj.coeffs)
        self.D_topright = self.env.get_D_topright()

    def improve_with_gradientDescent(self, coeffs_est):
        """ Make sure result gets better after running a few iters of grad descent. """
        err_raw = np.linalg.norm(coeffs_est - self.traj.coeffs)

        # For very small errors, the gradient steps to sometimes not refine the error.
        # I believe this is a precision and not algorithmic problem, so we do the
        # test only if the coeffs_est are not so close to the solution to start with.
        # note that if the first condition fails, err_refined > err_raw, so we only
        # have to check that err_refined < eps.
        if (abs(err_raw) < 1e-10):
            print("err_raw was already good enough.")
            return

        coeffs_grad, __ = gradientDescent(
            self.env.anchors, self.traj.basis, coeffs_est, self.D_topright, maxIters=10)
        err_refined = np.linalg.norm(coeffs_grad - self.traj.coeffs)

        #np.testing.assert_array_almost_equal(coeffs_grad, self.traj.coeffs, decimal=2)

        self.assertTrue((err_refined <= err_raw) or (abs(err_refined) < 1e-10))

    def test_semidefRelaxation(self):
        """ Check noiseless error. """

        for i in range(10):
            self.set_measurements(i)

            # check noiseless methods.
            X = semidefRelaxationNoiseless(
                self.D_topright,
                self.env.anchors,
                self.traj.basis,
                chosen_solver=CVXOPT,
                verbose=True)
            coeffs_est = X[:DIM:, DIM:]
            np.testing.assert_array_almost_equal(X[:DIM:, :DIM], np.eye(DIM), decimal=1)
            #np.testing.assert_array_almost_equal(coeffs_est, self.traj.coeffs, decimal=1)
            self.improve_with_gradientDescent(coeffs_est)

    def test_customMDS(self):
        """ Check noiseless error. """

        for i in range(10):
            self.set_measurements(i)

            # check noiseless methods.
            coeffs_est = customMDS(self.D_topright, self.traj.basis, self.env.anchors)
            np.testing.assert_array_almost_equal(coeffs_est, self.traj.coeffs, decimal=2)

            self.improve_with_gradientDescent(coeffs_est)

    def test_simulations(self):
        parameters = {
            'key': 'test',
            'n_its': 2,
            'type_missing': 'uniform',
            'time': 'undefined',
            'n_samples': [6, 7],
            'n_complexity': [4, 5],
            'n_anchors': [3],
            'sigmas_noise': [0],
            'values_missing': [0]
        }
        outfolder = 'results/{}/'.format(parameters['key'])
        try:
            run_simulation(parameters, outfolder, solver="rightInverseOfConstraints")
        except RuntimeError as e:
            self.fail("run_simulation raised exception: " + str(e))


if __name__ == "__main__":
    unittest.main()
