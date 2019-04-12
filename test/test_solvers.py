#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from common import test_prepare
test_prepare()

from cvxpy import CVXOPT
import numpy as np
import unittest

from solvers import *
from trajectory import Trajectory
from environment import Environment
from measurements import get_measurements


class TestSolvers(unittest.TestCase):
    def setUp(self):
        self.traj = Trajectory(n_complexity=5, dim=2)
        self.env = Environment(n_anchors=4)
        self.basis = []
        self.D_topright = []

    def set_measurements(self, seed=1):
        #  random trajectory and anchors
        self.traj.set_coeffs(seed=seed)
        self.env.set_random_anchors(seed=seed)

        # get measurements
        self.basis, self.D_topright = get_measurements(self.traj, self.env, seed=seed, n_samples=20)

    def test_semidefRelaxationNoiseless(self):
        """ Check noiseless error. """

        for i in range(10):

            self.set_measurements(seed=i)

            # check noiseless methods.
            X = semidefRelaxationNoiseless(
                self.D_topright, self.env.anchors, self.basis, chosen_solver=CVXOPT, verbose=False)
            coeffs_est = X[:DIM:, DIM:]
            np.testing.assert_array_almost_equal(X[:DIM:, :DIM], np.eye(DIM), decimal=1)
            np.testing.assert_array_almost_equal(coeffs_est, self.traj.coeffs, decimal=1)


if __name__ == "__main__":
    unittest.main()
