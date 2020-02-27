# -*- coding: utf-8 -*-

import sys
sys.path.append('../source')

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from scipy.io import loadmat
import seaborn as sns

from coordinate_fitting import fit_trajectory
from other_algorithms import apply_algorithm, error_measure, cost_function
from other_algorithms import pointwise_srls
from other_algorithms import get_grid, pointwise_rls
import probability as pr
from solvers import trajectory_recovery

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
    mask = (D)
    p = np.sort(np.sum(mask, axis=0))[::-1]
    return pr.full_rank_condition(list(p), dim + 1, K)


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
                                                     mse=mse,
                                                     cost_rls=None,
                                                     cost_srls=None)
    return points_fitted


def create_subsample(traj, D, times, anchors, n_measurements_list):
    """ Evaluate our algorithm and srls, srls for different numbers of measurements.

    Used to create Figure 7, second row, in Relax and Recover paper.
    
    """
    grid = get_grid(anchors, grid_size=0.5)
    results = pd.DataFrame(columns=['method', 'result', 'n_measurements'])
    num_seeds = 3
    for n_measurements in n_measurements_list:
        for seed in range(num_seeds):
            np.random.seed(seed)
            indices = np.random.choice(D.shape[0], n_measurements, replace=False)
            D_small = D[indices, :]

            times_small = np.array(times)[indices]
            basis_small = traj.get_basis(times=times_small)
            Chat = trajectory_recovery(D_small, anchors[:2, :], basis_small, weighted=True)

            results.loc[len(results), :] = dict(
                n_measurements=n_measurements,
                method='ours-weighted',
                result=Chat
            )

        p_rls, __ = pointwise_rls(D, anchors, traj, indices, grid)

        p_srls, __ = pointwise_srls(D, anchors, traj, indices)
        results.loc[len(results), :] = dict(
            n_measurements=n_measurements,
            method='srls',
            result=p_srls
        )
        results.loc[len(results), :] = dict(
            n_measurements=n_measurements,
            method='rls',
            result=p_rls
        )
    return results


def create_complexities(traj, D, times, anchors, list_complexities):
    """ Evaluate our algorithm and srls, srls for different complexities.

    Used to create Figure 7, first row, in Relax and Recover paper.
    
    """
    grid = get_grid(anchors, grid_size=0.5)
    results = pd.DataFrame(columns=['method', 'result', 'n_complexity'])
    num_seeds = 3
    for n_complexity in list_complexities:
        traj.set_n_complexity(n_complexity)

        basis = traj.get_basis(times=times)

        Chat = trajectory_recovery(D, anchors[:2, :], basis, weighted=True)

        results.loc[len(results), :] = dict(
            n_complexity=n_complexity,
            method='ours-weighted',
            result=Chat
        )

        indices = range(D.shape[0])[traj.dim + 2::3]
        p_rls, __ = pointwise_rls(D, anchors, traj, indices, grid)
        p_srls, __ = pointwise_srls(D, anchors, traj, indices)

        results.loc[len(results), :] = dict(
            n_complexity=n_complexity,
            method='srls',
            result=p_srls
        )
        results.loc[len(results), :] = dict(
            n_complexity=n_complexity,
            method='rls',
            result=p_rls
        )
    return results
