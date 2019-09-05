#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iterative_algorithms.py: Contains functions to build up the trajectory iteratively over time. 

"""

import numpy as np
import matplotlib.pylab as plt

from solvers import alternativePseudoInverse


def averaging_algorithm(D, anchors, basis, times, t_window=1.0):
    """ Iteratively compute estimates over fixed time window.

    :param D: measurement matrix with squared distances (N x M)
    :param anchors: anchor coordinates (dim x M)
    :param basis: basis vectors (dim x K)
    :param times: list of measurement times (length N)

    :param t_window: width of fixed time window. 

    """

    print(anchors.shape)
    line_counter = 0

    D_k = np.empty((0, D.shape[1]))
    basis_k = np.empty((basis.shape[0], 0))
    C_list = []
    t_list = []

    t_start = np.linspace(0, 2, 20)

    for t_s in t_start:
        tk = times[(times >= t_s) & (times < t_s + t_window)]
        for t_n in tk:
            idx = np.where(times == t_n)[0]

            d_mn_row = D[idx, :]
            f_n = basis[:, idx]

            D_k = np.r_[D_k, d_mn_row]
            basis_k = np.c_[basis_k, f_n]

        try:
            C_k = alternativePseudoInverse(D_k, anchors, basis_k)
            C_list.append(C_k)

        except AssertionError:
            print('skipping {:.2f} because only'.format(t_s), np.array(tk))

        D_k = np.empty((0, D.shape[1]))
        basis_k = np.empty((basis.shape[0], 0))
        t_list.append(tk)
    return C_list, t_list


def build_up_algorithm(D, anchors, basis, times, eps=1e-3):
    """ Build-up algorithm for trajectory estimation. 
    
    Build up different trajectories as long as measurements "fit". When they 
    stop fitting (see eps parameter), start a new trajectory.

    :param D: measurement matrix with squared distances (N x M)
    :param anchors: anchor coordinates (dim x M)
    :param basis: basis vectors (dim x K)
    :param times: list of measurement times (length N)

    :param eps: error threshold for starting new trajectory.

    """
    print(anchors.shape)
    C_k = None
    tk = []

    D_k = np.empty((0, D.shape[1]))
    basis_k = np.empty((basis.shape[0], 0))

    C_list = []
    t_list = []

    def g():
        r_n = C_k.dot(f_n).reshape((ams.shape[0], 1))
        dist_estimates = np.linalg.norm(ams - r_n, axis=0)
        return np.sum(dist_estimates**2 - d_mn_row)

    for d_mn_row, t_n, f_n in zip(D, times, basis.T):

        d_mn_row = d_mn_row.reshape((1, -1))
        ams = anchors[:, d_mn_row[0] > 0]
        ds = d_mn_row[0, d_mn_row[0] > 0]
        if C_k is None or g() <= eps:
            D_k = np.r_[D_k, d_mn_row]
            basis_k = np.c_[basis_k, f_n]
            tk.append(t_n)

            try:
                C_k = alternativePseudoInverse(D_k, anchors, basis_k)
            except AssertionError:
                print('skipping {:.2f} because only'.format(t_n), np.array(tk))
            #except Exception as e:
            #    raise e

        elif (C_k is not None) and g() > eps:
            print('changing to new trajectory, because error is {:.4f}'.format(g()))

            basis_k = f_n
            D_k = d_mn_row

            C_list.append(C_k)
            t_list.append(tk)

            tk = [t_n]
            C_k = None

    C_list.append(C_k)
    t_list.append(tk)

    return C_list, t_list
