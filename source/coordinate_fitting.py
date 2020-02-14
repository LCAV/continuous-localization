# -*- coding: utf-8 -*-
"""
coordinate_fitting.py: Fit the parametric trajectory to given number of coordinates.

"""

import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import minimize
from trajectory import Trajectory


def solve_for_coeffs(R, F):
    F_inv = np.linalg.inv(F.dot(F.T))
    return R.dot(F.T).dot(F_inv)


def solve_for_times(times, R, C, trajectory):
    def loss(times, R, C):
        F = trajectory.get_basis(times=times)
        return np.linalg.norm(R - C.dot(F))**2

    def grad_loss(times, R, C):
        F = trajectory.get_basis(times=times)
        F_prime = trajectory.get_basis_prime(times)

        grad_L_F = C.T.dot(R - C.dot(F))
        grad = np.diag(grad_L_F.T.dot(F_prime))
        return grad

    # Nelder-Mead seems to work better than BFGS, BFGS often gives a warning.
    res = minimize(loss, x0=times, args=(R, C), method='Nelder-Mead', options=dict(maxiter=200))
    #options=dict(xatol=1e-10, fatol=1e-10, maxiter=200))
    #res = minimize(loss, x0=times, args=(coordinates, coeffs), jac=grad_loss, method='BFGS')
    if not res.success:
        print('Warning: optimization did not succeed. Message of scipy.optimize.minimize:', res.message)
    return res.x


def fit_trajectory_and_times(coordinates, trajectory, max_iter=100, times=None):
    """ Fit a trajectory to positions (times and coefficients).

    :param coordinates: matrix of coordinates to fit trajectory to. Nxdim
    :param trajectory: Trajectory object.
    :param max_iter: max iterations.

    """
    N = coordinates.shape[1]
    if times is None:
        times = trajectory.get_times(N)
    K = trajectory.n_complexity
    d = trajectory.dim

    for i in range(max_iter):
        basis = trajectory.get_basis(times=times)
        assert basis.shape[0] == K
        assert basis.shape[1] == N
        coeffs = solve_for_coeffs(coordinates, basis)
        assert coeffs.shape[0] == d
        assert coeffs.shape[1] == K

        times = solve_for_times(times, coordinates, coeffs, trajectory)
    return coeffs, times


def fit_trajectory(coordinates, times, traj):
    """ Fit trajectory to positions (coefficients only). 
    
    :param coordinates: position coordinates (dim x N)
    :param times: list of corresponding times
    :param traj: Trajectory instance, of the model to be fitted.

    :return: fitted trajectory coefficients (dim x K)
    """
    basis = traj.get_basis(times=times)
    assert coordinates.shape[0] == traj.dim, coordinates.shape
    assert basis.shape[0] == traj.n_complexity
    assert basis.shape[1] == coordinates.shape[1], f'{basis.shape, coordinates.shape}'
    coeffs_hat = solve_for_coeffs(coordinates, basis)
    return np.array(coeffs_hat, dtype=np.float32)
