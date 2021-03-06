#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import common

import numpy as np
import unittest

from trajectory import Trajectory
from constraints import *
from measurements import get_measurements, create_anchors


class TestGeometry(unittest.TestCase):
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

    def test_constraints(self):
        """ Check the correct trajectory satisfies constraints.  """
        for i in range(100):
            self.set_measurements(i)

            # check the correct trajectory satisfies constraints
            e_ds, e_dprimes, deltas = get_constraints_identity(self.traj.n_complexity)
            for e_d, e_dprime, delta in zip(e_ds, e_dprimes, deltas):
                np.testing.assert_equal(e_d.T @ self.traj.Z_opt @ e_dprime, delta)

            t_mns, D_mns = get_extended_constraints(self.D_topright, self.anchors, self.basis)

            for t_mn, D_topright_mn in zip(t_mns, D_mns):
                t_mn = np.array(t_mn)
                np.testing.assert_almost_equal(t_mn.T @ self.traj.Z_opt @ t_mn, D_topright_mn)

                tmp = t_mn @ t_mn.T
                A = tmp.flatten()
                self.assertAlmostEqual(A @ (self.traj.Z_opt).flatten(), D_topright_mn)

            # test vectorized form of both constraints
            A, b = get_constraints_identity(self.traj.n_complexity, vectorized=True)
            np.testing.assert_array_almost_equal(A @ self.traj.Z_opt.flatten(), b)

            A, b = get_extended_constraints(self.D_topright, self.anchors, self.basis, vectorized=True, A=A, b=b)
            np.testing.assert_array_almost_equal(A @ self.traj.Z_opt.flatten(), b)

            A, b = get_constraints_symmetry(self.traj.n_complexity, vectorized=True)
            np.testing.assert_array_almost_equal(A @ self.traj.Z_opt.flatten(), b)


class TestGetFrame(unittest.TestCase):
    def test_dimensions(self):
        n_constrains = 5
        n_positions = 13
        self.assertEqual((n_constrains, n_positions), get_frame(n_constrains, n_positions).shape)


class TestGetLeftSubmatrix(unittest.TestCase):
    def test_dimensions(self):
        n_anchors = 3
        n_constrains = 5
        n_positions = 13
        ind_a = [0] * 8
        ind_b = ind_a
        anchors = create_anchors(2, n_anchors, check=True)
        frame = get_frame(n_constrains, n_positions)
        self.assertEqual((len(ind_a), (anchors.shape[0] + 1) * n_constrains),
                         get_left_submatrix(ind_a, ind_b, anchors, frame, extended=True).shape)
        self.assertEqual((len(ind_a), anchors.shape[0] * n_constrains),
                         get_left_submatrix(ind_a, ind_b, anchors, frame, extended=False).shape)


class TestGetRightSubmatrix(unittest.TestCase):
    def test_dimensions(self):
        n_constrains = 5
        n_positions = 13
        idx_f = [0] * 8
        frame = get_frame(n_constrains, n_positions)
        self.assertEqual((len(idx_f), n_constrains - 1), get_right_submatrix(idx_f, frame, reduced=True).shape)
        self.assertEqual((len(idx_f), n_constrains**2), get_right_submatrix(idx_f, frame, reduced=False).shape)


class TestGetConstraints(TestGeometry):

    def test_constraints_dimensions(self):
        self.set_measurements()
        T_A, T_B, b = get_constraints(self.D_topright, self.anchors, self.basis)
        first_dim = len(np.nonzero(self.D_topright)[0])
        self.assertEqual((first_dim, self.anchors.shape[0] * self.basis.shape[0]), T_A.shape)
        self.assertEqual((first_dim, self.basis.shape[0] * self.basis.shape[0]), T_B.shape)

    def test_C_constraints(self):
        for i in range(1):
            self.set_measurements(i)
            L = self.traj.coeffs.T.dot(self.traj.coeffs)
            T_A, T_B, b = get_constraints(self.D_topright, self.anchors, self.basis)
            T = np.c_[T_A, -T_B / 2]
            x = np.r_[self.traj.coeffs.flatten(), L.flatten()]

            np.testing.assert_array_almost_equal(T @ x, b)


if __name__ == "__main__":
    unittest.main()
