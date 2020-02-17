# -*- coding: utf-8 -*-
"""
other_algorithms.py: Baseline algorithms to compare against. 
"""

import pdb

import numpy as np
from scipy.optimize import least_squares

from pylocus.lateration import SRLS

from coordinate_fitting import fit_trajectory
from solvers import trajectory_recovery

EPS = 1e-10


def error_measure(points_gt, points_estimated, measure='mse'):
    """
    :param points_gt: ground truth positions (N x dim)
    :param points_estimated: estimated positions (N x dim)
    :param measure: method to use ('mae' or 'mse')
    """
    if (points_gt.shape[0] == 0) | (points_estimated.shape[0] == 0):
        return None
    assert points_gt.shape == points_estimated.shape, f'{points_gt.shape}, {points_estimated.shape}'

    if measure == 'mse':
        return np.mean((points_gt - points_estimated)**2)
    elif measure == 'mae':
        return np.mean(np.abs(points_gt - points_estimated))
    else:
        raise ValueError(measure)


def get_anchors_and_distances(D_sq, idx, dim=2):
    """ Get measurements for pointwise lateration. 

    Given squared distance matrix D and time index idx, 
    find all latest distance measurements up to this time index.

    :param D_sq: squared distance matrix (N x M)
    :param idx: time index for which we want measurements.

    :return: ndarray of distances squared (nx1) , 
             list (len n) of corresponding anchor indices.
    """
    assert idx >= 0 and idx < D_sq.shape[0]
    r2 = []
    anchors = []
    for a_id in range(D_sq.shape[1]):
        indices = np.where(D_sq[:idx + 1, a_id] > 0)[0]
        if len(indices) > 0:
            latest_idx = indices[-1]
            r2.append(D_sq[latest_idx, a_id])
            anchors.append(a_id)
    return np.array(r2).reshape((-1, 1)), np.array(anchors)


def init_lm(coeffs_real, method='ellipse', **kwargs):
    if 'ellipse' in method:
        coeffs = np.zeros(coeffs_real.shape)
        center = coeffs_real[:, 0]
        rx = np.max(coeffs_real[0, 1:]) - np.min(coeffs_real[0, 1:])
        ry = np.max(coeffs_real[1, 1:]) - np.min(coeffs_real[1, 1:])
        coeffs[0, 0] = center[0]
        coeffs[1, 0] = center[1]
        coeffs[0, 1] = rx
        coeffs[1, 2] = ry
        return coeffs
    elif 'line' in method:
        # 2D line with unit speed in 45 degrees, and 0 center.
        coeffs = np.zeros(coeffs_real.shape)
        center = [0, 0]
        v = np.sqrt(2)
        coeffs[0, 0] = center[0]
        coeffs[1, 0] = center[1]
        coeffs[0, 1] = v
        coeffs[1, 1] = v
        return coeffs
    elif 'noise' in method:
        sigma = kwargs.get('sigma', 0.1)
        return coeffs_real + np.random.normal(scale=sigma)
    elif 'real' in method:
        return coeffs_real
    else:
        raise ValueError(method)


def cost_function(C_vec, D_sq, A, F, squared=False):
    """ Return residuals of least squares distance error.

    :param C_vec: trajectory coefficients (length dim*K)
    :param D_sq: squared distance matrix (N x M)
    :param A: anchor coordinates (dim x M)
    :param F: trajectory basis functions (K x N)
    :param squared: if True, the distances in the cost function are squared. 

    :return: vector of residuals (length N)
    """
    dim = A.shape[0]
    C_k = C_vec.reshape((dim, -1))
    R = C_k.dot(F)
    diff = R[:, :, None] - A[:, None, :]  # dim x N x M
    D_est = np.linalg.norm(diff, axis=0)  # N x M

    # set the missing elements to zero.
    D_est[D_sq == 0.0] = 0.0
    if np.any(np.isnan(D_est)):
        raise ValueError('some nans in D_est')

    D = D_sq.copy()
    D[D > 0] = np.sqrt(D[D > 0])

    if squared:
        nonzero = (D**2 - D_est**2)[D > 0]
    else:
        nonzero = (D - D_est)[D > 0]
    cost = np.power(nonzero, 2).reshape((-1, ))
    return cost


def split_cost_function(X_vec, D_sq, A, F, squared=True):
    """ Return cost of distance squared, but with cost function split into coeffs and coeffs'coeffs:=L. Therefore the optimization variable is bigger but we only care about the first K*dim elements.

    :param X_vec: vector of trajectory coefficients and its squares, of length (dim*K+K*K)
    :param D_sq: squared distance matrix (N x M)
    :param A: anchor coordinates (dim x M)
    :param F: trajectory basis functions (K x N)
    :param squared: only here for constistency with cost_function. Has to be set to True or an error is raised.

    :return: vector of residuals.
    """

    if not squared:
        raise ValueError('Cannot split cost without squares.')

    dim = A.shape[0]
    K = F.shape[0]
    N, M = D_sq.shape
    assert A.shape[1] == M
    assert len(X_vec) == dim * K + K * K

    ns, ms = np.where(D_sq > 0)
    res = []

    # TODO(FD): below could be written without for loop,
    # but it is only used for testing so performance does not
    # matter.
    for n, m in zip(ns, ms):
        cost_n = 0.5 * (np.linalg.norm(A[:, m])**2 - D_sq[n, m])
        t_n = np.r_[np.outer(A[:, m], F[:, n]).reshape((-1, )), np.outer(F[:, n], F[:, n]).reshape((-1, ))]
        assert len(t_n) == len(X_vec)
        cost_n = cost_n - np.inner(t_n, X_vec)
        res.append(cost_n)
    return res


# TODO(FD) fix this function to pass unit tests.
def cost_jacobian(C_vec, D_sq, A, F, squared=True):
    """ Return Jacobian of squared distances cost function. 

    WARNING: this function does not pass its unit tests.

    :param C_vec: trajectory coefficients (dim x K)
    :param D_sq: squared distance matrix (N x M)
    :param A: anchor coordinates (dim x M)
    :param F: trajectory basis functions (K x N)
    :param squared: if True, the distances in the cost function are squared. Non-squared Jacobian not implemented yet.

    :return: (N x K*d) Jacobian matrix.
    """
    if not squared:
        raise NotImplementedError('cost_jacobian for non-squared distances')

    l = cost_function(C_vec, D_sq, A, F, squared=True)  # cost vector (N)
    ns, ms = np.where(D_sq > 0)

    N = len(l)
    Kd = len(C_vec)
    dim = A.shape[0]
    K = int(Kd / dim)

    jacobian = np.empty((N, Kd))  # N x Kd

    C_k = C_vec.reshape((dim, -1))
    R = C_k.dot(F)
    for j, (l_n, m_n, n) in enumerate(zip(l, ms, ns)):
        f_n = F[:, n]
        assert len(f_n) == K

        # factor is the derivative of the norm squared with respect to coeffs matrix.
        factor = -2 * (A[:, m_n] - C_k.dot(f_n)).reshape((dim, 1)).dot(f_n.reshape((1, -1)))  # dim x K
        if (np.abs(l_n) > EPS):
            jacobian_mat = -2 * np.sqrt(np.abs(l_n)) * factor
        else:
            jacobian_mat = np.zeros((dim, K))
        jacobian[j, :] = jacobian_mat.reshape((-1, ))
    if np.any(np.isnan(jacobian)):
        print('Problems in cost_jacobian. Going in debugging mode.')
        pdb.set_trace()
    return jacobian


def least_squares_lm(D, anchors, basis, x0, verbose=False, cost='simple', jacobian=False):
    """ Solve using Levenberg Marquardt. 
    
    :param cost: Cost function to use, can be either:
        - 'squared': squared distances
        - 'simple': non-squared distances
        - 'split': split the cost in coeffs'coeffs=L and coeffs, optimize for whole thing at once.
    """
    dim = anchors.shape[0]
    M = anchors.shape[1]
    K = basis.shape[0]
    N = basis.shape[1]
    assert D.shape == (N, M), D.shape
    assert len(x0) == dim * K, f'{len(x0)}!={dim}*{K}'

    if jacobian:
        print(
            'Warning: the analytical jacobian will be passed to the least squares solver, but it has not passed all tests yet. This might lead to unexpected behavior.'
        )

    if np.any(np.isnan(x0)):
        raise ValueError(f'invalid x0 {x0}')

    scipy_verbose = 2 if verbose else 0

    if (cost == 'squared') and jacobian:
        res = least_squares(cost_function,
                            jac=cost_jacobian,
                            x0=x0,
                            method='lm',
                            args=(D, anchors, basis),
                            kwargs={'squared': True},
                            verbose=scipy_verbose)  # xtol=1e-20, ftol=1e-10,
    elif (cost == 'squared') and (not jacobian):
        res = least_squares(cost_function,
                            x0=x0,
                            method='lm',
                            args=(D, anchors, basis),
                            kwargs={'squared': True},
                            verbose=scipy_verbose)  # xtol=1e-20, ftol=1e-10,

    elif (cost == 'simple') and (not jacobian):
        res = least_squares(cost_function,
                            x0=x0,
                            method='lm',
                            args=(D, anchors, basis),
                            kwargs={'squared': False},
                            verbose=scipy_verbose)  # xtol=1e-20, ftol=1e-10,
    elif (cost == 'simple') and jacobian:
        raise NotImplementedError('Cannot do Jacobian without squares.')
    elif cost == 'split':
        C = x0.reshape((dim, K))
        L = C.T.dot(C)
        x0_extended = np.r_[x0, L.reshape((-1, ))]
        res = least_squares(split_cost_function,
                            x0=x0_extended,
                            method='lm',
                            args=(D, anchors, basis),
                            verbose=scipy_verbose)  # xtol=1e-20, ftol=1e-10,

    if not res.success:
        if verbose:
            print('LM failed with message:', res.message)
        return None
    if res.success:
        if verbose:
            print('LM succeeded with message:', res.message)

        # We only need to take the first K*dim elements
        # because for cost=='split' there are more. But this
        # doesn't hurt for the others.
        return res.x[:dim * K].reshape((dim, K))


def get_grid(anchors, grid_size=1.0):
    x_range, y_range = np.array([np.min(anchors, axis=1), np.max(anchors, axis=1)]).T
    xx, yy = np.meshgrid(
        np.arange(*x_range, step=grid_size),
        np.arange(*y_range, step=grid_size),
    )
    grid = np.c_[xx.flatten(), yy.flatten()]
    return grid


def RLS(anchors, r2, grid, interpolation='nearest'):
    """ Get RLS estimate.

    :param r2: list of measured distances (length M)
    :param anchors: anchor coordinates (M x dim)
    :param grid: grid coordinates (N x dim)

    :param interpolation: How to interpolate.
      - 'nearest' (default): return minimum grid point.
    """
    assert len(r2) == anchors.shape[0]
    assert grid.shape[1] == anchors.shape[1]
    r2 = r2.flatten()

    D_estimated = np.linalg.norm(anchors[None, :, :] - grid[:, None, :], axis=2)  # N x M
    cost = np.sum((D_estimated - np.sqrt(r2[None, :]))**2, axis=1)
    if interpolation == 'nearest':
        argmin = np.argmin(cost)
        return grid[argmin, :]


def pointwise_lateration(D, anchors, traj, indices, method='srls', grid=None):
    """ Solve using point-wise lateration. 

    :param indices: points at which we want to compute SRLS.
    :param method: Method to use. Currently supported:
        - 'rls': Range Least-Squares (need to give grid)
        - 'srls': SRLS
    :param grid: coordinates of grid for RLS(N_grid x dim) 

    :return: points, valid_indices
      - points: coordinates of shape (N x dim)
      - valid_indices: vector of corresponding indices. 
    """

    assert anchors.shape[0] == traj.dim
    assert anchors.shape[1] == D.shape[1], f'{anchors.shape}, {D.shape}'

    points = []
    valid_indices = []
    for idx in indices:
        r2, a_indices = get_anchors_and_distances(D, idx)

        # too many measurements
        if len(r2) > traj.dim + 2:
            #print(f'SRLS: too many measurements available! choosing random subset of {traj.dim + 2}')
            choice = np.random.choice(len(r2), traj.dim + 2, replace=False)
            r2 = r2[choice]
            a_indices = a_indices[choice]
            assert len(r2) == traj.dim + 2
            assert len(a_indices) == traj.dim + 2

        # too few measurements
        elif len(r2) < traj.dim + 2:
            #print('SRLS: skipping {} cause not enough measurements'.format(idx))
            continue

        anchors_here = anchors[:2, a_indices].T  #N x d
        weights = np.ones(r2.shape)

        if method == 'srls':
            estimate = SRLS(anchors_here, weights, r2)
        elif method == 'rls':
            estimate = RLS(anchors_here, r2, grid=grid)
        else:
            raise ValueError(method)

        points.append(estimate)
        valid_indices.append(idx)
    return np.array(points), valid_indices


def pointwise_srls(D, anchors, traj, indices):
    return pointwise_lateration(D, anchors, traj, indices, method='srls', grid=None)


def pointwise_rls(D, anchors, traj, indices, grid):
    return pointwise_lateration(D, anchors, traj, indices, method='rls', grid=grid)


def apply_algorithm(traj, D, times, anchors, method='ours'):
    """
    Apply method to given measurements.

    :return: Chat, points, indices
        Chat contains the trajectory fitted to the measurements.
        points 

    """
    if method == 'ours-weighted':
        basis = traj.get_basis(times=times)
        Chat = trajectory_recovery(D, anchors, basis, weighted=True)
        return Chat, None, None
    elif method == 'ours':
        basis = traj.get_basis(times=times)
        Chat = trajectory_recovery(D, anchors, basis, weighted=False)
        return Chat, None, None
    elif method == 'srls':
        indices = range(D.shape[0])[traj.dim + 2::3]
        points, indices = pointwise_srls(D, anchors, traj, indices)
        times = np.array(times)[indices]
        Chat = None
        if points.shape[0] >= traj.n_complexity:
            Chat = fit_trajectory(points.T, times=times, traj=traj)
        else:
            print(f'Warning in apply_algorithm(srls): cannot fit trajectory to points of shape {points.shape}.')
        return Chat, points, indices
    elif method == 'rls':
        indices = range(D.shape[0])[traj.dim + 2::3]
        grid = get_grid(anchors, grid_size=0.5)
        points, indices = pointwise_rls(D, anchors, traj, indices, grid=grid)
        times = np.array(times)[indices]
        Chat = None
        if points.shape[0] >= traj.n_complexity:
            Chat = fit_trajectory(points.T, times=times, traj=traj)
        else:
            print(f'Warning in apply_algorithm(rls): cannot fit trajectory to points of shape {points.shape}.')
        return Chat, points, indices
    elif method == 'lm-ellipse':
        basis = traj.get_basis(times=times)
        c0 = init_lm(traj.coeffs, method='ellipse').flatten()
        Chat = least_squares_lm(D, anchors, basis, c0, cost='simple', jacobian=False)
        return Chat, None, None
    elif method == 'lm-line':
        basis = traj.get_basis(times=times)
        c0 = init_lm(traj.coeffs, method='line').flatten()
        Chat = least_squares_lm(D, anchors, basis, c0, cost='simple', jacobian=False)
        return Chat, None, None
    elif method == 'lm-ours-weighted':
        basis = traj.get_basis(times=times)
        c0 = trajectory_recovery(D, anchors, basis, weighted=True)
        Chat = None
        if c0 is not None:
            c0 = c0.flatten()
            Chat = least_squares_lm(D, anchors, basis, c0, cost='simple', jacobian=False)
        return Chat, None, None
    else:
        raise ValueError(method)
