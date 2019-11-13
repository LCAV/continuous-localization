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


def cost_function(C_vec, D_sq, A, F, squared=False, verbose=False):
    """ Return cost of distance.

    :param C_vec: trajectory coefficients (1 x dim*K)
    :param D: squared distance matrix (N x M)
    :param A: anchor coordinates (dim x M)
    :param F: trajectory basis functions (K x N)

    :return: vector of residuals.
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


def split_cost_function(X_vec, D_sq, A, F, squared=False, verbose=False):
    """ Return cost of distance squared, but with cost function split into C and C'C:=L. Therefore the optimization variable is bigger but we only care about the first K*dim elements.

    :param X_vec: vector of trajectory coefficients and its squares, of length (dim*K+K*K)
    :param D: squared distance matrix (N x M)
    :param A: anchor coordinates (dim x M)
    :param F: trajectory basis functions (K x N)

    :return: vector of residuals.
    """
    dim = A.shape[0]
    K = F.shape[0]
    N, M = D_sq.shape
    assert A.shape[1] == M
    assert len(X_vec) == dim * K + K * K

    ns, ms = np.where(D_sq > 0)
    res = []

    # TODO(FD): below could be written without for loop,
    # but it is only used for testing so performance does not
    # matter and I didn't want to spend too much time figuring it out.
    for n, m in zip(ns, ms):
        cost_n = 0.5 * (np.linalg.norm(A[:, m])**2 - D_sq[n, m])
        t_n = np.r_[np.outer(A[:, m], F[:, n]).reshape((-1, )), np.outer(F[:, n], F[:, n]).reshape((-1, ))]
        assert len(t_n) == len(X_vec)
        cost_n = cost_n - np.inner(t_n, X_vec)
        res.append(cost_n)
    return res


def cost_jacobian(C_vec, D, A, F, squared=True, verbose=False):
    """ Return maximum likelihood cost of distances.

    :param C_k: trajectory coefficients (dim x K)
    :param D: squared distance matrix (N x M)
    :param A: anchor coordinates (dim x M)
    :param F: trajectory basis functions (K x N)

    :return: (N x K*d) Jacobian matrix.
    """
    if not squared:
        raise NotImplementedError('cost_jacobian for non-squared distances')

    l = cost_function(C_vec, D, A, F)  # cost vector (N)
    ns, ms = get_m_n(D)

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


def least_squares_lm(D, anchors, basis, x0, verbose=False, cost='simple', jacobian=False):
    """ Solve using Levenberg Marquardt. 
    
    :param cost: Cost function to use, can be either:
        - 'squared': squared distances
        - 'simple': non-squared distances
        - 'split': split the cost in C'C=L and C, optimize for whole thing at once.
    """
    dim = anchors.shape[0]
    M = anchors.shape[1]
    K = basis.shape[0]
    N = basis.shape[1]
    assert D.shape == (N, M), D.shape
    assert len(x0) == dim * K, f'{len(x0)}!={dim}*{K}'

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
