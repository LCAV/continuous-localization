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


class TestSolvers(unittest.TestCase):
    def setUp(self):
        self.traj = Trajectory()
        self.env = Environment()
        self.D_topright = []

    def set_measurements(self, seed=1):
        self.traj.set_trajectory(seed=seed)
        self.env.set_random_anchors(seed=seed)
        self.env.set_D(self.traj)
        self.D_topright = self.env.D[:self.traj.n_positions, self.traj.n_positions:]

    def test_semidefRelaxationNoiseless(self):
        """ Check noiseless error. """

        for i in range(10):
            self.set_measurements(i)

            # check noiseless methods.
            X = semidefRelaxationNoiseless(
                self.D_topright,
                self.env.anchors,
                self.traj.basis,
                chosen_solver=CVXOPT,
                verbose=False)
            coeffs_est = X[:DIM:, DIM:]
            np.testing.assert_array_almost_equal(X[:DIM:, :DIM], np.eye(DIM), decimal=1)
            np.testing.assert_array_almost_equal(coeffs_est, self.traj.coeffs, decimal=1)


if __name__ == "__main__":
    unittest.main()
