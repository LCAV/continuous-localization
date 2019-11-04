#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import common

from cvxpy import CVXOPT
import numpy as np
import unittest

from solvers import *
from trajectory import Trajectory
from measurements import get_measurements, create_anchors


class TestSolvers(unittest.TestCase):
    def setUp(self):
        self.n_anchors = 4
        self.traj = Trajectory(n_complexity=5, dim=2)
        self.basis = []
        self.D_topright = []

    def set_measurements(self, seed=1):
        #  random trajectory and anchors
        self.traj.set_coeffs(seed=seed)
        np.random.seed(seed)
        self.anchors = create_anchors(self.traj.dim, self.n_anchors)

        # get measurements
        self.basis, self.D_topright = get_measurements(self.traj, self.anchors, seed=seed, n_samples=20)

    def test_semidefRelaxationNoiseless(self):
        """ Check noiseless error. """

        for i in range(10):

            self.set_measurements(seed=i)

            # check noiseless methods.
            X = semidefRelaxationNoiseless(self.D_topright,
                                           self.anchors,
                                           self.basis,
                                           chosen_solver=CVXOPT,
                                           verbose=False)
            coeffs_est = X[:DIM:, DIM:]
            np.testing.assert_array_almost_equal(X[:DIM:, :DIM], np.eye(DIM), decimal=1)
            np.testing.assert_array_almost_equal(coeffs_est, self.traj.coeffs, decimal=1)


if __name__ == "__main__":
    unittest.main()
