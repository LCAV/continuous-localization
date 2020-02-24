#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_results_polynomial.py: Generate polynomial results (average over many linear movements).
"""

import sys
from os.path import abspath, dirname
this_dir = dirname(abspath(__file__))
sys.path.append(this_dir + '/../source/')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate_dataset import compute_distance_matrix, compute_anchors, calibrate
from public_data_utils import read_dataset, get_plotting_params, get_ground_truth, TIME_RANGES
from results_generation import generate_results, generate_suitable_mask, add_gt_fitting
from simulation import arg_parser

METHODS = ['ours-weighted', 'ours', 'lm-ours-weighted', 'lm-line', 'srls', 'rls']

if __name__ == "__main__":
    ##### Initialization  #####
    np.random.seed(1)

    dataset_file = this_dir + '/../datasets/Plaza1.mat'
    list_complexities = [2]
    list_measurements = [8, 10, 20, 30, 40, 50, 60]
    anchor_names = None  # use all anchors.
    verbose = True
    chosen_distance = 'distance'  # distance to use (can also be _calib, or _gt)
    range_system_id = 'Range'

    #Parameters used for polynomial results from paper.
    #outfile = this_dir + '/../results/polynomial_tuesday.pkl'
    #total_n_it = 20

    description = 'Generate polynomial reconstruction results.'
    outfile, plotting, total_n_it = arg_parser(description=description)

    full_df, anchors_df, traj = read_dataset(dataset_file, verbose=False)
    xlim, ylim = get_plotting_params(dataset_file)

    assert range_system_id in full_df.system_id.unique(), full_df.system_id.unique()

    # extract the piecewise linear time ranges.
    max_time = full_df.timestamp.max()
    time_ranges = [t for t in TIME_RANGES if t[0] < max_time]

    mask = np.array([False] * len(full_df))
    for time_range in time_ranges:
        mask = mask | ((full_df.timestamp > time_range[0]) & (full_df.timestamp < time_range[1])).values
    full_df = full_df[mask]

    if chosen_distance == 'distance_calib':
        calibrate(full_df)

    ##### Bring data in correct form #####
    anchors = compute_anchors(anchors_df, anchor_names)
    n_min = 2 * (traj.dim + 2) - 1
    print(f'need at least {n_min} measurements.')

    ##### Run experiments #####

    result_df = pd.DataFrame()

    if plotting:
        fig, axs = plt.subplots(1, len(time_ranges), sharex=True, sharey=True)
        fig.set_size_inches(1 * len(time_ranges), 2)

    for k, time_range in enumerate(time_ranges):
        if verbose:
            print('time range', time_range)

        ## Filter measurements
        part_df = full_df[(full_df.timestamp < time_range[1]) & (full_df.timestamp > time_range[0])]
        times = part_df[part_df.system_id == range_system_id].timestamp.unique()
        D, times = compute_distance_matrix(part_df, anchors_df, anchor_names, times, chosen_distance)
        if np.sum(D > 0) > D.shape[0]:
            print(f'Warning: multiple measurements for times:{np.sum(np.sum(D > 0, axis=1) > 1)}/{D.shape[0]}!')
        elif np.sum(D > 0) < n_min:
            print('Condition (7) not satisfied!')
            continue

        ## Construct ground truth.
        points_gt = get_ground_truth(part_df, times)
        points_gt = points_gt.loc[:, ['px', 'py']].values

        ## Plot all
        if plotting:
            ax = axs[k]
            ax.scatter(*points_gt.T, color='black', s=1.0)
            ax.set_title(f'{time_range[0]}-')

        for n_measurements in list_measurements:
            if verbose:
                print(f'n_measurements={n_measurements}')
            if n_measurements > D.shape[0]:
                continue

            for n_complexity in list_complexities:
                if verbose:
                    print(f'K={n_complexity}')
                traj.set_n_complexity(n_complexity)
                n_min = traj.n_complexity * (traj.dim + 2) - 1
                if n_measurements < n_min:
                    print(f'K={n_complexity}: skipping {n_measurements} < {n_min}')
                    continue

                for n_it in range(total_n_it):
                    indices = generate_suitable_mask(D, traj.dim, traj.n_complexity, n_measurements)
                    D_small = D[indices, :]

                    times_small = np.array(times)[indices]
                    points_small = points_gt[indices, :]

                    df = generate_results(traj, D_small, times_small, anchors, points_small, methods=METHODS, n_it=k)
                    points_fitted = add_gt_fitting(traj, times_small, points_small, df, n_it=k)

                    result_df = pd.concat((result_df, df), ignore_index=True, sort=False)
                if outfile != '':
                    result_df.to_pickle(outfile)
                    print('saved as', outfile)

        ## Plot the last one.
        if not plotting:
            continue

        traj_plot = traj.copy()
        for method, df_method in df.groupby('method'):
            coeffs, __ = df_method.loc[:, 'plotting'].values[0]
            if coeffs is not None:
                traj_plot.set_coeffs(coeffs=coeffs)
                traj_plot.plot_pretty(ax=ax, times=times, label=method)

    if plotting:
        axs[-1].legend(loc='lower right')
        [ax.set_xlim(*xlim) for ax in axs]
        [ax.set_ylim(*ylim) for ax in axs]
        plt.show()
