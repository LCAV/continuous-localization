#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_results_bandlimited.py: Generate bandlimited results.
"""

import sys
sys.path.append('../source/')

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

from evaluate_dataset import compute_distance_matrix, compute_anchors
from results_generation import generate_results, add_gt_fitting, generate_suitable_mask
from plotting_tools import plot_complexities, add_scalebar
from public_data_utils import read_dataset, get_ground_truth, get_plotting_params

METHODS = ['ours-weighted', 'ours', 'lm-ellipse', 'lm-ours-weighted', 'srls', 'rls']

if __name__ == "__main__":
    ##### Initialization  #####
    np.random.seed(1)

    filename = '../datasets/Plaza2.mat'  # triangle

    resultname = '../results/bandlimited_tuesday.pkl'
    resultname = '../results/test.pkl'

    full_df, anchors_df, traj = read_dataset(filename)
    xlim, ylim = get_plotting_params(filename)

    chosen_distance = 'distance'
    #chosen_distance = 'distance_gt'
    range_system_id = 'Range'
    assert range_system_id in full_df.system_id.unique()

    list_complexities = [3, 5, 11, 19]
    list_measurements = [40, 100, 200, 300, 400, 499]
    total_n_it = 20
    anchor_names = None  # use all anchors.

    plotting = True
    verbose = True

    ##### Bring data in correct form #####
    anchors = compute_anchors(anchors_df, anchor_names)
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

    result_df = pd.DataFrame(columns=['n_it', 'n_complexity', 'n_measurements', 'mae', 'mse', 'method', 'plotting'])
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
                indices = generate_suitable_mask(D, traj.dim, traj.n_complexity, n_measurements)

                D_small = D[indices, :]
                times_small = np.array(times)[indices]
                points_small = points_gt[indices, :]

                current_results = generate_results(traj,
                                                   D_small,
                                                   times_small,
                                                   anchors,
                                                   points_small,
                                                   n_it=n_it,
                                                   methods=METHODS)
                points_fitted = add_gt_fitting(traj, times_small, points_small, current_results, n_it=0)

                result_df = pd.concat((result_df, current_results), ignore_index=True, sort=False)

            if resultname != '':
                result_df.to_pickle(resultname)
                print('saved as', resultname)

            if plotting:
                results_plotting = {}
                for m, df in current_results.groupby('method'):
                    results_plotting[m] = df.iloc[-1].loc['plotting']
                ax = axs[i, j]
                ax = plot_complexities(traj, times_small, results_plotting, points_fitted, ax)

    if plotting:
        fig.set_size_inches(*fig_size)
        add_scalebar(axs[0, 0], 20, loc='lower left')
        [ax.set_xlim(*xlim) for ax in axs.flatten()]
        [ax.set_ylim(*ylim) for ax in axs.flatten()]
        plt.show()
    print('done')
