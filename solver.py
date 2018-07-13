#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2018 Frederike Duembgen <frederike.duembgen@gmail.com>
#
# Distributed under terms of the MIT license.

from global_variables import DIM
import numpy as np

"""
solver.py: 
"""

def get_constraints_D(D, anchors, basis, linear=False, A=None, b=None):
    n_positions,__ = D.shape
    W = D > 0
    Ns, Ms = np.where(W)

    if not linear:
        t_mns = []
        D_mns = []
    elif A is None and b is None:
        A = []
        b = []

    for i, (m, n) in enumerate(zip(Ms, Ns)):
        e_n = np.zeros((n_positions, 1))
        e_n[n] = 1.0
        a_m = np.reshape(anchors[:, m], (-1, 1))
        t_mn = np.r_[a_m, -basis @ e_n]

        if linear:
            t_mn = np.array(t_mn)
            tmp = t_mn @ t_mn.T
            A.append(tmp.flatten())
            b.append(D[n, m])
        else:
            t_mns.append(t_mn)
            D_mns.append(D[n, m])

    if not linear:
        return t_mns, D_mns
    else:
        return A, b


def get_constraints_identity(n_complexity, linear=False, A=None, b=None):
    if not linear:
        e_ds = []
        e_dprimes = []
        deltas = []
    elif A is None and b is None:
        A = []
        b = []

    for d in range(DIM):
        e_d = np.zeros((DIM + n_complexity, 1))
        e_d[d] = 1.0

        for dprime in range(DIM):
            e_dprime = np.zeros((DIM + n_complexity, 1))
            e_dprime[dprime] = 1.0

            delta = 1.0 if d == dprime else 0.0

            if linear:
                tmp = e_dprime @ e_d.T
                A.append(tmp.flatten())
                b.append(delta)
            else:
                e_ds.append(e_d)
                e_dprimes.append(e_dprime)
                deltas.append(delta)

    if not linear:
        return e_ds, e_dprimes, deltas
    else:
        return A, b


def get_constraints_symmetry(n_complexity, linear=True, A=None, b=None):
    if linear and A is None:
        A = []
    if linear and b is None:
        b = []

    for i in range(DIM + n_complexity):
        for j in range(i+1, DIM + n_complexity):
            tmp = np.zeros((DIM + n_complexity)*(DIM + n_complexity))
            tmp[i*(DIM + n_complexity)+j] = 1
            tmp[j*(DIM + n_complexity)+i] = -1
            if linear:
                A.append(tmp.flatten())
                b.append(0)
    return A, b


def get_constraints_matrix(D_topright, anchors, basis):
    A = []
    b = []

    n_complexity = basis.shape[0]

    get_constraints_D(D_topright, anchors, basis, linear=True, A=A, b=b)
    TmnStack = np.array(A)
    u, s, vh = np.linalg.svd(TmnStack)
    print('number of nonzero elements of s:', len(s[s > 1e-10]))
    #Think we need (K+D)*(K+D+1)/2-(D*(D+1)/2)=K(K+1)/2+DK=18
    print('need:', n_complexity * (n_complexity + 1) / 2 + DIM * n_complexity)

    get_constraints_identity(n_complexity, linear=True, A=A, b=b)
    get_constraints_symmetry(n_complexity, linear=True, A=A, b=b)

    A = np.array(A)
    b = np.array(b)
    return A, b
