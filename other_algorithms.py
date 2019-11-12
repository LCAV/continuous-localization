#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
other_algorithms.py: Baseline algorithms to compare against. 
"""
import pdb

import numpy as np
from scipy.optimize import least_squares

from pylocus.lateration import SRLS

EPS = 1e-10


def calculate_error(Chat, C, error_type='MAE'):
    """ Return error measure between C and Chat. """
    if error_type == 'MAE':
        return np.mean(np.abs(Chat - C))
    else:
        NotImplementedError(error_type)


def get_anchors_and_distances(D, idx, dim=2):
    """ Get measurements for pointwise lateration. 

    Given squared distance matrix D and time index idx, 
    find all latest distance measurements up to this time index.

    :param D: squared distance matrix (N x M)
    :param idx: time index for which we want measurements.

    :return: ndarray of distances squared (nx1) , 
             list (len n) of corresponding anchor indices.
    """
    assert idx >= 0 and idx < D.shape[0]
    r2 = []
    anchors = []
    counter = 0
    for a_id in range(D.shape[1]):
        indices = np.where(D[:idx + 1, a_id] > 0)[0]
        if len(indices) > 0:
            latest_idx = indices[-1]
            r2.append(D[latest_idx, a_id])
            anchors.append(a_id)
            counter += 1
            if counter > dim + 1:
                break  # enough measurements for lateration.
    return np.array(r2).reshape((-1, 1)), anchors


def get_m_n(D):
    """ Get the anchor indices of the flattened distance measurements.
    """
    n, m = np.where(D > 0)
    return np.array(n), np.array(m)


def cost_function(C_k_vec, D_sq, A, F, verbose=False):
    """ Return cost of distance squared.

    :param C_k: trajectory coefficients (dim x K)
    :param D_sq: squared distance matrix (N x M)
    :param A: anchor coordinates (dim x M)
    :param F: trajectory basis functions (K x N)

    :return: vector of residuals.
    """
    dim = A.shape[0]
    C_k = C_k_vec.reshape((dim, -1))
    R = C_k.dot(F)
    diff = R[:, :, None] - A[:, None, :]  # dim x N x M
    D_est_sq = np.linalg.norm(diff, axis=0)**2  # N x M

    # set the missing elements to zero.
    D_est_sq[D_sq == 0.0] = 0.0
    if np.any(np.isnan(D_est_sq)):
        raise ValueError('some nans in D_est_sq')

    nonzero = (D_sq - D_est_sq)[D_sq > 0]
    cost = np.power(nonzero, 2).reshape((-1, ))
    return cost


def cost_jacobian(C_k_vec, D, A, F, verbose=False):
    """ Return maximum likelihood cost of distances.

    :param C_k: trajectory coefficients (dim x K)
    :param D: squared distance matrix (N x M)
    :param A: anchor coordinates (dim x M)
    :param F: trajectory basis functions (K x N)

    :return: (N x K*d) Jacobian matrix.
    """
    l = cost_function(C_k_vec, D, A, F)  # cost vector (N)
    ns, ms = get_m_n(D)

    N = len(l)
    Kd = len(C_k_vec)
    dim = A.shape[0]
    K = int(Kd / dim)

    jacobian = np.empty((N, Kd))  # N x Kd

    C_k = C_k_vec.reshape((dim, -1))
    R = C_k.dot(F)
    for j, (l_n, m_n, n) in enumerate(zip(l, ms, ns)):
        f_n = F[:, n]
        assert len(f_n) == K

        # factor is the derivative of the norm squared with respect to C matrix.
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


def least_squares_lm(D, anchors, basis, x0, verbose=False, squares=True, jacobian=False):
    """ Solve using Levenberg Marquardt. """

    if np.any(np.isnan(x0)):
        raise ValueError(f'invalid x0 {x0}')

    scipy_verbose = 2 if verbose else 0

    if squares and jacobian:
        res = least_squares(cost_function,
                            jac=cost_jacobian,
                            x0=x0,
                            method='lm',
                            args=(D, anchors, basis),
                            x_scale='jac',
                            verbose=scipy_verbose)  # xtol=1e-20, ftol=1e-10,
    elif squares and (not jacobian):
        res = least_squares(cost_function,
                            x0=x0,
                            method='lm',
                            args=(D, anchors, basis),
                            x_scale='jac',
                            verbose=scipy_verbose)  # xtol=1e-20, ftol=1e-10,

    elif (not squares) and (not jacobian):
        res = least_squares(cost_function,
                            jac=cost_jacobian,
                            x0=x0,
                            method='lm',
                            args=(D, anchors, basis),
                            x_scale='jac',
                            verbose=scipy_verbose)  # xtol=1e-20, ftol=1e-10,
    elif (not squares) and jacobian:
        raise NotImplementedError('Cannot do Jacobian without squares.')
    if not res.success:
        if verbose:
            print('LM failed with message:', res.message)
        return None
    if res.success:
        if verbose:
            print('LM succeeded with message:', res.message)
        return res.x.reshape((2, -1))


def pointwise_srls(D, anchors, basis, traj, indices):
    """ Solve using point-wise SRLS. """
    points = []
    for idx in indices[::traj.dim + 1]:
        r2, a_indices = get_anchors_and_distances(D, idx)
        if len(r2) > traj.dim + 1:
            anchors_srls = anchors[:2, a_indices].T  #N x d
            weights = np.ones(r2.shape)
            points.append(SRLS(anchors_srls, weights, r2))
        else:
            print('SRLS: skipping {} cause not enough measurements'.format(idx))
    return points
