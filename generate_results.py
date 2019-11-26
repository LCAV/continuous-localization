#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from scipy.io import loadmat
import seaborn as sns

from fit_curve import fit_trajectory
from other_algorithms import apply_algorithm, error_measure, cost_function

METHODS = ['ours-weighted', 'ours', 'lm-ellipse', 'lm-ours-weighted', 'srls', 'rls']


def generate_suitable_mask(D, dim, K, n_measurements):
    counter = 0
    while counter < 100:
        indices = sorted(np.random.choice(D.shape[0], n_measurements, replace=False))
        D_small = D[indices, :]
        if test_hypothesis(D_small, dim, K):
            return indices
        counter += 1
    raise ValueError('Did not find suitable mask after 100.')


def test_hypothesis(D, dim, K):
    import hypothesis as h
    mask = (D)
    p = np.sort(np.sum(mask, axis=0))[::-1]
    return h.limit_condition(list(p), dim + 1, K)


def generate_results(traj, D_small, times_small, anchors, points_small, methods=METHODS, n_it=0):
    n_complexity = traj.n_complexity
    n_measurements = np.sum(D_small > 0)
    current_results = pd.DataFrame(
        columns=['n_it', 'n_complexity', 'n_measurements', 'mae', 'mse', 'method', 'plotting', 'cost_rls', 'cost_srls'])

    basis_small = traj.get_basis(times=times_small)

    for method in methods:
        C_hat, p_hat, lat_idx = apply_algorithm(traj, D_small, times_small, anchors, method=method)
        plotting = (C_hat, p_hat)
        mae = mse = cost_rls = cost_slrs = None
        if C_hat is not None:
            traj.set_coeffs(coeffs=C_hat)
            p_fitted = traj.get_sampling_points(times=times_small).T
            mae = error_measure(p_fitted, points_small, 'mae')
            mse = error_measure(p_fitted, points_small, 'mse')
            cost_rls = np.sum(cost_function(C_hat.reshape((-1, )), D_small, anchors, basis_small, squared=False))
            cost_srls = np.sum(cost_function(C_hat.reshape((-1, )), D_small, anchors, basis_small, squared=True))
        current_results.loc[len(current_results)] = dict(plotting=plotting,
                                                         n_complexity=n_complexity,
                                                         n_measurements=n_measurements,
                                                         method=method,
                                                         n_it=n_it,
                                                         mae=mae,
                                                         mse=mse,
                                                         cost_rls=cost_rls,
                                                         cost_srls=cost_srls)

        # do raw version if applicable
        if method in ['rls', 'srls']:
            points_small_lat = points_small[lat_idx]
            mae = error_measure(p_hat, points_small_lat, 'mae')
            mse = error_measure(p_hat, points_small_lat, 'mse')
            current_results.loc[len(current_results)] = dict(plotting=(None, None),
                                                             n_complexity=n_complexity,
                                                             n_measurements=n_measurements,
                                                             method=method + ' raw',
                                                             n_it=n_it,
                                                             mae=mae,
                                                             mse=mse,
                                                             cost_rls=cost_rls,
                                                             cost_srls=cost_srls)
    return current_results


def add_gt_fitting(traj, times_small, points_small, current_results, n_it=0):
    # fit ground truth to chosen points.
    n_complexity = traj.n_complexity
    n_measurements = len(times_small)

    coeffs = fit_trajectory(points_small.T, times=times_small, traj=traj)
    traj_gt = traj.copy()
    traj_gt.set_coeffs(coeffs=coeffs)
    points_fitted = traj_gt.get_sampling_points(times=times_small).T

    mse = error_measure(points_fitted, points_small, 'mse')
    mae = error_measure(points_fitted, points_small, 'mae')
    current_results.loc[len(current_results)] = dict(plotting=(coeffs, points_fitted),
                                                     n_complexity=n_complexity,
                                                     n_measurements=n_measurements,
                                                     method='gt',
                                                     n_it=n_it,
                                                     mae=mae,
                                                     mse=mse)
    return points_fitted
