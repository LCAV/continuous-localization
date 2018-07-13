# coding: utf-8

# In[4]:

import numpy as np
import unittest

from SampTrajsTools import *
from trajectory import Trajectory
from environment import Environment
from solver import *


class TestTrajectory(unittest.TestCase):
    def setUp(self):
        self.traj = Trajectory()
        self.env = Environment()

    def test_trajectory(self):
        """ Check the correct trajectory is PSD.  """

        for i in range(100):
            self.traj.set_trajectory(seed=i)

            w, v = np.linalg.eig(self.traj.Z_opt)
            self.assertTrue(np.all(w > -1e-10))

    def test_constraints(self):
        """ Check the correct trajectory satisfies constraints.  """

        for i in range(1):
            self.traj.set_trajectory(seed=i)
            self.env.set_random_anchors(seed=i)
            self.env.set_D(self.traj)
            D_topright = self.env.D[:self.traj.n_positions,
                                    self.traj.n_positions:]

            #check the correct trajectory satisfies constraints

            e_ds, e_dprimes, deltas = get_constraints_identity(
                self.traj.n_complexity)
            for e_d, e_dprime, delta in zip(e_ds, e_dprimes, deltas):
                np.testing.assert_equal(
                    e_d.T @ self.traj.Z_opt @ e_dprime, delta)

            t_mns, D_mns = get_constraints_D(
                D_topright, self.env.anchors, self.traj.basis)

            for t_mn, D_topright_mn in zip(t_mns, D_mns):
                t_mn = np.array(t_mn)
                np.testing.assert_almost_equal(
                    t_mn.T @ self.traj.Z_opt @ t_mn, D_topright_mn)

            A, b = get_constraints_identity(
                self.traj.n_complexity, linear=True)
            np.testing.assert_array_almost_equal(
                A @ self.traj.Z_opt.flatten(), b)

            A, b = get_constraints_D(
                D_topright, self.env.anchors, self.traj.basis, linear=True)
            np.testing.assert_array_almost_equal(
                A @ self.traj.Z_opt.flatten(), b)

            A, b = get_constraints_symmetry(
                self.traj.n_complexity, linear=True)
            np.testing.assert_array_almost_equal(
                A @ self.traj.Z_opt.flatten(), b)

    def test_all_linear(self):
        self.traj.set_trajectory()
        self.env.set_random_anchors()
        self.env.set_D(self.traj)
        D_topright = self.env.D[:self.traj.n_positions, self.traj.n_positions:]

        A, b = get_constraints_matrix(
            D_topright, self.env.anchors, self.traj.basis)
        np.testing.assert_array_almost_equal(A @ self.traj.Z_opt.flatten(), b)


if __name__ == "__main__":
    unittest.main()
