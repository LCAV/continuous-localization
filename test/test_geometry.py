#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from os.path import abspath, dirname
print(__file__)
sys.path.append(abspath(dirname(__file__) + '/../'))
print(sys.path)

import numpy as np
import unittest

from trajectory import Trajectory
from environment import Environment
from constraints import *


class TestGeometry(unittest.TestCase):
    def setUp(self):
        self.traj = Trajectory()
        self.env = Environment()

        times = self.traj.get_times()
        self.traj.set_basis(times)

    def test_constraints(self, seed=1):
        """ Check the correct trajectory satisfies constraints.  """

        self.traj.set_coeffs(seed=seed)
        self.env.set_random_anchors(seed=seed)

        self.env.set_D(self.traj.coeffs)
        Z_opt = self.traj.get_Z_opt()
        D_topright = self.env.get_D_topright()

        #check the correct trajectory satisfies constraints

        e_ds, e_dprimes, deltas = get_constraints_identity(self.traj.n_complexity)
        for e_d, e_dprime, delta in zip(e_ds, e_dprimes, deltas):
            np.testing.assert_equal(e_d.T @ Z_opt @ e_dprime, delta)

        t_mns, D_mns = get_constraints_D(D_topright, self.env.anchors, self.traj.basis)

        for t_mn, D_topright_mn in zip(t_mns, D_mns):
            t_mn = np.array(t_mn)
            np.testing.assert_almost_equal(t_mn.T @ Z_opt @ t_mn, D_topright_mn)

            tmp = t_mn @ t_mn.T
            A = tmp.flatten()
            self.assertAlmostEqual(A @ Z_opt.flatten(), D_topright_mn)

        # test linear form of both constraints

        A, b = get_constraints_identity(self.traj.n_complexity, linear=True)
        np.testing.assert_array_almost_equal(A @ Z_opt.flatten(), b)

        A, b = get_constraints_D(
            D_topright, self.env.anchors, self.traj.basis, linear=True, A=A, b=b)
        np.testing.assert_array_almost_equal(A @ Z_opt.flatten(), b)

        A, b = get_constraints_symmetry(self.traj.n_complexity, linear=True)
        np.testing.assert_array_almost_equal(A @ Z_opt.flatten(), b)

    def test_all_linear(self, seed=1):
        self.traj.set_coeffs(seed=seed)
        self.env.set_random_anchors(seed=seed)
        self.env.set_D(self.traj.coeffs)
        Z_opt = self.traj.get_Z_opt()
        D_topright = self.env.get_D_topright()

        A, b = get_constraints_matrix(D_topright, self.env.anchors, self.traj.basis)
        np.testing.assert_array_almost_equal(A @ Z_opt.flatten(), b)

    def notest_many(self):
        for i in range(100):
            self.test_constraints(self, i)
            self.test_all_linear(self, i)


if __name__ == "__main__":
    unittest.main()
