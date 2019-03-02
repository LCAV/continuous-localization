#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from common import test_prepare
test_prepare()

import numpy as np
import unittest

from trajectory import Trajectory
from environment import Environment
from constraints import *


class TestGeometry(unittest.TestCase):
    def setUp(self):
        self.traj = Trajectory()
        self.env = Environment()

    def test_constraints(self):
        """ Check the correct trajectory satisfies constraints.  """
        for i in range(100):
            self.traj.set_trajectory(seed=i)
            self.env.set_random_anchors(seed=i)
            self.env.set_D(self.traj)
            D_topright = self.env.D[:self.traj.n_positions, self.traj.n_positions:]

            #check the correct trajectory satisfies constraints

            e_ds, e_dprimes, deltas = get_constraints_identity(self.traj.n_complexity)
            for e_d, e_dprime, delta in zip(e_ds, e_dprimes, deltas):
                np.testing.assert_equal(e_d.T @ self.traj.Z_opt @ e_dprime, delta)

            t_mns, D_mns = get_constraints_D(D_topright, self.env.anchors, self.traj.basis)

            for t_mn, D_topright_mn in zip(t_mns, D_mns):
                t_mn = np.array(t_mn)
                np.testing.assert_almost_equal(t_mn.T @ self.traj.Z_opt @ t_mn, D_topright_mn)

                tmp = t_mn @ t_mn.T
                A = tmp.flatten()
                self.assertAlmostEqual(A @ (self.traj.Z_opt).flatten(), D_topright_mn)

            # test vectorized form of both constraints
            A, b = get_constraints_identity(self.traj.n_complexity, vectorized=True)
            np.testing.assert_array_almost_equal(A @ self.traj.Z_opt.flatten(), b)

            A, b = get_constraints_D(
                D_topright, self.env.anchors, self.traj.basis, vectorized=True, A=A, b=b)
            np.testing.assert_array_almost_equal(A @ self.traj.Z_opt.flatten(), b)

            A, b = get_constraints_symmetry(self.traj.n_complexity, vectorized=True)
            np.testing.assert_array_almost_equal(A @ self.traj.Z_opt.flatten(), b)

    def test_all_vectorized(self):
        self.traj.set_trajectory()
        self.env.set_random_anchors()
        self.env.set_D(self.traj)
        D_topright = self.env.D[:self.traj.n_positions, self.traj.n_positions:]

        A, b = get_constraints_matrix(D_topright, self.env.anchors, self.traj.basis)
        np.testing.assert_array_almost_equal(A @ self.traj.Z_opt.flatten(), b)

    def test_C_constraints(self):
        self.traj.set_trajectory()
        self.env.set_random_anchors()
        self.env.set_D(self.traj)
        D_topright = self.env.D[:self.traj.n_positions, self.traj.n_positions:]

        L = self.traj.coeffs.T.dot(self.traj.coeffs)

        T_A, T_B, b = get_C_constraints(D_topright, self.env.anchors, self.traj.basis)
        T = np.c_[T_A, -T_B / 2]
        x = np.r_[self.traj.coeffs.flatten(), L.flatten()]

        np.testing.assert_array_almost_equal(T @ x, b)


if __name__ == "__main__":
    unittest.main()
