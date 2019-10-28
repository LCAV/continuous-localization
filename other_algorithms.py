#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
other_algorithms.py: Baseline algorithms to compare against. 
"""
import numpy as np
from scipy.optimize import least_squares

from pylocus.lateration import SRLS


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


def cost_function(C_k_vec, D, A, F, verbose=False):
    """ Return maximum likelihood cost of distances.

    :param C_k: trajectory coefficients (dim x K)
    :param D: squared distance matrix (N x M)
    :param A: anchor coordinates (dim x M)
    :param F: trajectory basis functions (K x N)

    :return: vector of residuals.
    """
    C_k = C_k_vec.reshape((2, -1))
    R = C_k.dot(F)
    if verbose:
        print('R  dim x N', R.shape)  # dim x N
    diff = R[:, :, None] - A[:, None, :]
    if verbose:
        print('diff  dim x N x M', diff.shape)
    D_est = np.linalg.norm(diff, axis=0)  #  dim x N x M
    if verbose:
        print('N x M', D_est.shape, D.shape)
    D_est[D == 0.0] = 0.0
    assert not np.any(np.isnan(D_est))
    D[D > 0] = np.sqrt(D[D > 0])
    nonzero = (D_est - D)[D > 0]
    return np.power(nonzero, 2).reshape((-1, ))


def least_squares_lm(D, anchors, basis, x0):
    """ Solve using Levenerg Marquardt. """
    res = least_squares(cost_function, x0=x0, method='lm', args=(D, anchors[:2], basis))
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
