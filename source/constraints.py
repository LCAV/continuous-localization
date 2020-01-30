#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
constraints.py: Generate constraints for semidefinite relaxations. 

This module can be used to generate constraints on Z and on coeffs.

For Z: 
    Z = [I_D coeffs; coeffs^T L]

"""

from global_variables import DIM
import numpy as np


def verify_dimensions(D_topright, anchors, basis):
    """
    :param D_topright: n_positions x n_anchors
    :param anchors: dim x n_anchors
    :param basis: n_complexity x n_positions

    """
    n_positions, n_anchors = D_topright.shape
    n_complexity = basis.shape[0]
    dim = anchors.shape[0]

    assert n_positions > n_complexity, 'Cannot compute {} coeffs with only {} measurements.'.format(
        n_complexity, n_positions)
    assert n_anchors > dim, 'Cannot localize in {}D with only {} anchors.'.format(dim, n_anchors)

    assert basis.shape[1] == n_positions, basis.shape
    assert anchors.shape[1] == n_anchors, anchors.shape


def get_constraints_D(D_topright, anchors, basis, vectorized=False, A=None, b=None):
    """ Get constraints on Z given by the distances.

    .. math::
        t_{mn}^T Z t_{mn} = d_{mn}^2

    where :math:`t_{mn} = (a_m^T -f_n^T)^T`

    :param D_topright: squared distsance measurements, shape (n_positions x n_anchors)
    :param anchors: anchor coordinates, shape (dim x n_anchors)
    :param basis: basis vectors, shape (n_complexity x n_positions)

    """
    verify_dimensions(D_topright, anchors, basis)

    dim = anchors.shape[0]
    Ns, Ms = np.where(D_topright > 0)
    n_complexity = basis.shape[0]

    if not vectorized:
        t_mns = []
        D_mns = []
    elif A is None and b is None:
        A = []
        b = []

    for i, (m, n) in enumerate(zip(Ms, Ns)):
        a_m = np.reshape(anchors[:, m], (-1, 1))
        f_n = basis[:, n].reshape(n_complexity, 1)
        t_mn = np.r_[a_m, -f_n]

        if vectorized:
            t_mn = np.array(t_mn)
            tmp = t_mn @ t_mn.T
            A.append(tmp.flatten())
            b.append(D_topright[n, m])
        else:
            t_mns.append(t_mn)
            D_mns.append(D_topright[n, m])

    if not vectorized:
        return t_mns, D_mns
    else:
        return A, b


def get_constraints_identity(n_complexity, dim=DIM, vectorized=False, A=None, b=None):
    """ Get identity constraints for top left of Z matrix. 

    for not vectorized:

    .. math::
        e_d Z e_{d'} = \delta_{dd'}

    for vectorized: 

    .. math::
        vect(e_{d'}e_{d}^T)  vect(Z) = \delta_{dd'}

    :param A: if given, we append the constraints to A. 
    :param b: if given, we append the constraints to b. 

    """

    if (A is not None or b is not None) and not vectorized:
        raise NotImplementedError

    if not vectorized:
        e_ds = []
        e_dprimes = []
        deltas = []
    elif A is None and b is None:
        A = []
        b = []

    for d in range(dim):
        e_d = np.zeros((dim + n_complexity, 1))
        e_d[d] = 1.0

        for dprime in range(dim):
            e_dprime = np.zeros((dim + n_complexity, 1))
            e_dprime[dprime] = 1.0

            delta = 1.0 if d == dprime else 0.0

            if vectorized:
                tmp = e_dprime @ e_d.T
                A.append(tmp.flatten())
                b.append(delta)
            else:
                e_ds.append(e_d)
                e_dprimes.append(e_dprime)
                deltas.append(delta)

    if not vectorized:
        return e_ds, e_dprimes, deltas
    else:
        return A, b


def get_constraints_symmetry(n_complexity, dim=DIM, vectorized=True, A=None, b=None):
    if not vectorized:
        NotImplementedError
    if vectorized and A is None:
        A = []
    if vectorized and b is None:
        b = []

    for i in range(dim + n_complexity):
        for j in range(i + 1, dim + n_complexity):
            tmp = np.zeros((dim + n_complexity) * (dim + n_complexity))
            tmp[i * (dim + n_complexity) + j] = 1
            tmp[j * (dim + n_complexity) + i] = -1
            if vectorized:
                A.append(tmp.flatten())
                b.append(0)
    return A, b


def get_C_constraints(D_topright, anchors, basis, weighted=False):
    """ Return constraints TA, TB, and vector b as defined in paper.

    :param D_topright: matrix of square distances, of shape n_positions x n_anchors.
    :param weighted: bool, if true return measurements and constraints divided by the weight depended on the distance, in order to normalise errors. Makes sense only when errors are added to distances
    """

    verify_dimensions(D_topright, anchors, basis)

    Ns, Ms = np.where(D_topright > 0)
    dim = basis.shape[0]

    T_A = []
    T_B = []
    b = []

    for (m, n) in zip(Ms, Ns):
        weight = 1.0 / np.sqrt((D_topright[n, m]) + 1e-1) if weighted else 1.0
        a_m = np.reshape(anchors[:, m], (-1, 1))
        f_n = basis[:, n].reshape(dim, 1)

        tmp = a_m @ f_n.T
        T_A.append(weight * tmp.flatten())

        tmp = f_n @ f_n.T
        T_B.append(weight * tmp.flatten())

        b.append(weight * (np.sum(a_m * a_m) - D_topright[n, m]) / 2)

    T_A = np.array(T_A)
    T_B = np.array(T_B)
    b = np.array(b)

    return T_A, T_B, b
