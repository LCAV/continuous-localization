#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fit_curve.py: Fit our model to a trajectory given in coordinates.
"""

import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import minimize
from trajectory import Trajectory


def solve_for_C(R, F):
    F_inv = np.linalg.inv(F.dot(F.T))
    return R.dot(F.T).dot(F_inv)


def solve_for_times(times, R, C, trajectory):
    def L(times, R, C):
        F = trajectory.get_basis(times=times)
        return np.linalg.norm(R - C.dot(F))**2

    def grad_L(times, R, C):
        F = trajectory.get_basis(times=times)
        F_prime = trajectory.get_basis_prime(times)

        grad_L_F = C.T.dot(R - C.dot(F))
        grad = np.diag(grad_L_F.T.dot(F_prime))
        return grad

    # Nelder-Mead seems to work better than BFGS, BFGS often gives a warning.
    res = minimize(L, x0=times, args=(R, C), method='Nelder-Mead')
    #options=dict(xatol=1e-10, fatol=1e-10, maxiter=200))
    #res = minimize(L, x0=times, args=(R, C), jac=grad_L, method='BFGS')
    if not res.success:
        print('Warning: optimization did not succeed. Message of scipy.optimize.minimize:', res.message)
    return res.x


def fit_trajectory_and_times(R, trajectory, max_iter=100, times=None):
    """ Fit a trajectory to positions (times and coefficients).

    :param R: matrix of coordinates to fit trajectory to. Nxdim
    :param trajectory: Trajectory object.
    :param max_iter: max iterations.

    """
    N = R.shape[1]
    if times is None:
        times = trajectory.get_times(N)

    K = trajectory.n_complexity
    d = trajectory.dim

    for i in range(max_iter):
        F = trajectory.get_basis(times=times)
        assert F.shape[0] == K
        assert F.shape[1] == N
        C = solve_for_C(R, F)
        assert C.shape[0] == d
        assert C.shape[1] == K

        times = solve_for_times(times, R, C, trajectory)
    return C, times


def fit_trajectory(R, times, traj):
    """ Fit trajectory to positions (coefficients only). 
    
    :param R: position coordinates (dim x N)
    :param times: list of corresponding times
    :param traj: Trajectory instance, of the model to be fitted.

    :return: fitted trajectory coefficients (dim x K)
    """
    F = traj.get_basis(times=times)
    assert R.shape[0] == traj.dim
    assert F.shape[0] == traj.n_complexity
    assert F.shape[1] == R.shape[1]
    Chat = solve_for_C(R, F)
    return np.array(Chat, dtype=np.float32)


if __name__ == "__main__":
    n_complexity = 11
    dim = 2
    model = 'full_bandlimited'
    trajectory = Trajectory(n_complexity=n_complexity, dim=dim, model=model, full_period=True)

    # load real coordinates (choose one)
    # fnames = ['fitting/circle', 'fitting/complicated', 'fitting/plaza2']
    fnames = ['fitting/plaza2']
    for fname in fnames:
        coords_original = np.loadtxt(fname + '.txt', delimiter=',')

        # create a new trajectory
        new_trajectory = trajectory.copy()

        # fit coefficients of new trajectory
        C, times = fit_trajectory_and_times(coords_original, new_trajectory, max_iter=10)
        new_trajectory.set_coeffs(coeffs=C)

        # plot results
        basis = new_trajectory.get_basis(times=times)
        coords_reconstructed = new_trajectory.get_sampling_points(basis=basis)
        plt.figure()
        plt.title('Trajectory fitting result')
        plt.plot(*coords_original, color='green', label='original1')
        plt.plot(*coords_reconstructed, color='green', linestyle=':', label='reconstructed1')
        plt.legend()
        plt.savefig(fname + '_fit.png')
        print('saved as', fname + '_fit.png')
        plt.show()
