#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from scipy.io import loadmat
import seaborn as sns

#from data_utils import *
from data_utils import prepare_dataset, read_dataset, get_coordinates, get_ground_truth, get_plotting_params
from evaluate_dataset import compute_distance_matrix
from fit_curve import fit_trajectory
from other_algorithms import pointwise_srls, apply_algorithm, error_measure
from plotting_tools import plot_complexities, add_scalebar
from trajectory_creator import get_trajectory


def test_hypothesis(D, dim, K):
    import hypothesis as h
    mask = (D)
    p = np.sort(np.sum(mask, axis=0))[::-1]
    assert h.limit_condition(list(p), dim + 1, K)


def add_measurement(result_df, method=''):
    result_df.loc[len(result_df)] = dict(n_complexity=n_complexity,
                                         n_measurements=n_measurements,
                                         method=method,
                                         n_it=n_it,
                                         mae=mae,
                                         mse=mse)


if __name__ == "__main__":
    ##### Initialization  #####
    np.random.seed(1)

    #filename = 'datasets/uah1.mat' # fingers
    #filename = 'datasets/Plaza1.mat'; # zig zag.
    #filename = 'datasets/Plaza2.mat'  # triangle
    #filename = 'datasets/Gesling1.mat'  #
    filename = 'datasets/Gesling2.mat'  #

    resultname = 'results/algorithms_Gesling1.pkl'

    full_df, anchors_df, traj = read_dataset(filename)
    xlim, ylim = get_plotting_params(filename)

    chosen_distance = 'distance'
    range_system_id = 'Range'
    assert range_system_id in full_df.system_id.unique()
    #chosen_distance = 'distance_gt'

    list_complexities = [3, 5]  #, 11, 19]
    list_measurements = [40, 100]  #, 200, 300, 400, 499]
    methods = ['ours-weighted', 'ours']
    methods += ['lm-ellipse', 'lm-ours']
    methods += ['srls', 'rls']
    total_n_it = 2  #0
    anchor_names = None  # use all anchors.

    plotting = True
    verbose = True

    ##### Bring data in correct form #####
    anchors = get_coordinates(anchors_df, anchor_names)
    times = full_df[full_df.system_id == range_system_id].timestamp.unique()
    D, times = compute_distance_matrix(full_df, anchors_df, anchor_names, times, chosen_distance)
    if np.sum(D > 0) > D.shape[0]:
        print('Warning: multiple measurements for times:{}/{}!'.format(np.sum(np.sum(D > 0, axis=1) > 1), D.shape[0]))
    points_gt = get_ground_truth(full_df, times).values
    anchors = anchors[:2, :]

    ##### Run experiments #####
    if plotting:
        fig, axs = plt.subplots(len(list_measurements), len(list_complexities), sharex=True, sharey=True)
        fig_size = [5, 1.2 * len(list_measurements)]

    result_df = pd.DataFrame(columns=['n_it', 'n_complexity', 'n_measurements', 'mae', 'mse', 'method'])
    for j, n_complexity in enumerate(list_complexities):
        if verbose:
            print(f'K={n_complexity}')
        traj.set_n_complexity(n_complexity)

        if plotting:
            axs[0, j].set_title(f'K={n_complexity}')

        for i, n_measurements in enumerate(list_measurements):
            if plotting:
                axs[i, 0].set_ylabel(f'N={n_measurements}')
            if verbose:
                print(f'n_measurements={n_measurements}')

            for n_it in range(total_n_it):
                indices = sorted(np.random.choice(D.shape[0], n_measurements, replace=False))
                D_small = D[indices, :]

                # test hypothesis
                test_hypothesis(D_small, traj.dim, traj.n_complexity)

                times_small = np.array(times)[indices]
                basis_small = traj.get_basis(times=times_small)
                points_small = points_gt[indices, :]

                results = {}
                for method in methods:
                    C_hat, p_hat, lat_idx = apply_algorithm(traj, D_small, times_small, anchors, method=method)
                    results[method] = (C_hat, p_hat)
                    traj.set_coeffs(coeffs=C_hat)
                    p_fitted = traj.get_sampling_points(times=times_small).T
                    mae = error_measure(p_fitted, points_small, 'mae')
                    mse = error_measure(p_fitted, points_small, 'mse')
                    add_measurement(result_df, method=method)

                    # do raw version if applicable
                    if method in ['rls', 'srls']:
                        points_lat = points_small[lat_idx]
                        mae = error_measure(p_hat, points_lat, 'mae')
                        mse = error_measure(p_hat, points_lat, 'mse')
                        add_measurement(result_df, method=method + ' raw')
                # fit ground truth to chosen points.
                coeffs = fit_trajectory(points_small.T, times=times_small, traj=traj)
                traj_gt = traj.copy()
                traj_gt.set_coeffs(coeffs=coeffs)
                points_fitted = traj_gt.get_sampling_points(times=times_small).T

                mse = error_measure(points_fitted, points_small, 'mse')
                mae = error_measure(points_fitted, points_small, 'mae')
                add_measurement(result_df, 'gt')

            if resultname != '':
                result_df.to_pickle(resultname)
                print('saved as', resultname)

            if plotting:
                ax = axs[i, j]
                ax = plot_complexities(traj, times_small, results, points_fitted, ax)

    if plotting:
        fig.set_size_inches(*fig_size)
        add_scalebar(axs[0, 0], 20, loc='lower left')
        [ax.set_xlim(*xlim) for ax in axs.flatten()]
        [ax.set_ylim(*ylim) for ax in axs.flatten()]
        plt.show()
    print('done')
