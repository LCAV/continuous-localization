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

        self.traj.get_random_trajectory(seed=1)

        w, v = np.linalg.eig(self.traj.Z_opt)
        # TODO make test
        print('should be positive:')
        print(w)

    def test_constraints(self):
        """ Check the correct trajectory satisfies constraints.  """

        for i in range(1):
            self.traj.get_random_trajectory(seed=i)
            self.env.get_random_anchors(seed=i)
            self.env.get_measurements(self.traj)

            #check the correct trajectory satisfies constraints

            e_ds, e_dprimes, deltas = get_constraints_identity(self.traj.n_complexity)
            for e_d, e_dprime, delta in zip(e_ds, e_dprimes, deltas):
                np.testing.assert_equal(e_d.T @ self.traj.Z_opt @ e_dprime, delta)

            t_mns, D_mns = get_constraints_D(
                self.env.D_topright, self.env.anchors, self.traj.basis)

            for t_mn, D_topright_mn in zip(t_mns, D_mns):
                t_mn = np.array(t_mn)
                np.testing.assert_almost_equal(t_mn.T @ self.traj.Z_opt @ t_mn, D_topright_mn)

            A, b = get_constraints_identity(
                self.traj.n_complexity, linear=True)
            np.testing.assert_array_almost_equal(A @ self.traj.Z_opt.flatten(), b)

            A, b = get_constraints_D(
                self.env.D_topright, self.env.anchors, self.traj.basis, linear=True)
            np.testing.assert_array_almost_equal(A @ self.traj.Z_opt.flatten(), b)

            A, b = get_constraints_symmetry(
                self.traj.n_complexity, linear=True)
            np.testing.assert_array_almost_equal(A @ self.traj.Z_opt.flatten(), b)

    def test_all_linear(self): 
        self.traj.get_random_trajectory()
        self.env.get_random_anchors()
        self.env.get_measurements(self.traj)

        A, b = get_constraints_matrix(
            self.env.D_topright, self.env.anchors, self.traj.basis)
        np.testing.assert_array_almost_equal(A @ self.traj.Z_opt.flatten(), b)

    #  def test_nullspace(self):
        # TODO decide what to check here. 
        #  DIM = 2

        #  print(ConstraintsMat.shape)
        #  print(ConstraintsVec.shape)
        #  u, s, vh = np.linalg.svd(ConstraintsMat, full_matrices=True)

        #  print(np.around(s, 3))

        #  #construct right inverse and check i
        #  num_zero_SVs = len(np.where(s < 1e-10)[0])
        #  Z_hat = vh[:-num_zero_SVs, :].T@np.diag(1/s[:-num_zero_SVs])@u[:, :len(
            #  s)-num_zero_SVs].T@ConstraintsVec  # right inverse
        #  Z_hat = Z_hat.reshape([DIM + n_complexity, DIM + n_complexity])
        #  # should satisfy constraints since it's a right inverse
        #  print(np.isclose(ConstraintsMat @ Z_hat.flatten(), ConstraintsVec))
        #  coeffs_hat = Z_hat[:DIM, DIM:]
        #  print(np.isclose(coeffs, coeffs_hat))

        #  print('find basis vectors of null space')
        #  tmp = vh[-num_zero_SVs:, :]
        #  print(tmp.shape)
        #  nullSpace = []
        #  for i in range(num_zero_SVs):
            #  nullSpace.append(tmp[i, :].reshape(
                #  [DIM + n_complexity, DIM + n_complexity]))

        #  nullSpace = np.array(nullSpace)
        #  Z_hat2 = Z_hat + nullSpace[0, :] + 2 * \
            #  nullSpace[1, :] + 3*nullSpace[2, :]
        #  print(np.isclose(ConstraintsMat@(
            #  Z_hat2.flatten()), ConstraintsVec))
        #  print(np.around(nullSpace[0, :], 5))
        #  print(np.around(nullSpace[1, :], 5))
        #  print(np.around(nullSpace[2, :], 5))

if __name__ == "__main__":
    unittest.main()
