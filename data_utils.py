#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
import seaborn as sns

from evaluate_dataset import format_anchors_df, format_data_df
from evaluate_dataset import add_gt_raw, apply_distance_gt
from iterative_algorithms import build_up_algorithm
from iterative_algorithms import averaging_algorithm
from plotting_tools import savefig
from solvers import alternativePseudoInverse
from trajectory_creator import get_trajectory

# Need to give different systems a name.
gt_system_id = "GT"
range_system_id = "Range"
gt_anchor_id = "GT"


def create_anchors_df(anchors_data):
    """ Create standard anchors dataframe. 

    :param anchors_data: anchors data read from .mat file ('TL' field).
    """
    anchors_df = pd.DataFrame(columns=['anchor_id', 'system_id', 'px', 'py', 'pz'])
    anchor_ids = np.unique(anchor_data[:, 0])
    for i, anchor_id in enumerate(anchor_ids):
        anchors_df.loc[i, 'anchor_id'] = anchor_id
        anchors_df.loc[i, 'system_id'] = range_system_id

        # it is weird that there is more than one value for each anchor, it looks
        # like this was a bug in the dataset. we make sure they are all
        # the same and pick the first.
        px_values = np.unique(anchor_data[anchor_data[:, 0] == anchor_id, 1])
        py_values = np.unique(anchor_data[anchor_data[:, 0] == anchor_id, 2])
        assert len(px_values) == 1
        assert len(py_values) == 1
        anchors_df.loc[i, 'px'] = px_values[0]
        anchors_df.loc[i, 'py'] = py_values[0]

    return anchors_df


def create_full_df(range_data, gt_data):
    """" Create full dataframe. """
    range_df = pd.DataFrame(columns=['timestamp', 'px', 'py', 'pz', 'distance', 'system_id', 'anchor_id'])
    range_df.loc[:, 'distance'] = range_data[:, 3]
    range_df.loc[:, 'timestamp'] = range_data[:, 0]
    range_df.loc[:, 'anchor_id'] = range_data[:, 2]
    range_df.loc[:, 'system_id'] = range_system_id

    gt_df = pd.DataFrame(columns=range_df.columns)
    gt_df.loc[:, 'px'] = result_dict['GT'][:, 1]
    gt_df.loc[:, 'py'] = result_dict['GT'][:, 2]
    gt_df.loc[:, 'timestamp'] = result_dict['GT'][:, 0]
    gt_df.loc[:, 'anchor_id'] = gt_anchor_id
    gt_df.loc[:, 'system_id'] = gt_system_id

    full_df = pd.concat([range_df, gt_df], ignore_index=True)
    full_df.sort_values('timestamp', inplace=True)
    full_df.reset_index(drop=True, inplace=True)
    full_df.loc[:, 'timestamp'] = full_df.timestamp - full_df.timestamp.min()
    return full_df


def get_ground_truth(full_df, times):
    """ Find one ground truth for each time when we have a distance measurement. 
    """
    ground_truth_pos = full_df.loc[full_df.timestamp.isin(times), ['timestamp', 'px', 'py', 'pz']]
    ground_truth_pos = ground_truth_pos.astype(np.float32)
    ground_truth_pos = ground_truth_pos.groupby('timestamp').agg(np.nanmean)
    ground_truth_pos.reset_index(inplace=True)
    return ground_truth_pos


def get_coordinates(anchors_df, anchor_names):
    """ Sort anchors according to names.  """
    anchors_df = anchors_df.set_index('anchor_name')
    anchors_df = anchors_df.loc[anchor_names]
    anchors_df.reset_index(drop=False, inplace=True)
    anchors = anchors_df.loc[:, ['px', 'py', 'pz']].values.astype(np.float32).T
    return anchors


def get_smooth_points(C_list, t_list, traj):
    """ Average the obtained trajectories. """
    result_df = pd.DataFrame(columns=['px', 'py', 't'])
    for Chat, t in zip(C_list, t_list):
        traj.set_coeffs(coeffs=Chat)
        positions = traj.get_sampling_points(times=t)
        this_df = pd.DataFrame({'px': positions[0, :], 'py': positions[1, :], 't': t})
        result_df = pd.concat((this_df, result_df))
    result_df.sort_values('t', inplace=True)
    result_df.reindex()

    import datetime
    mean_window = 10
    datetimes = [datetime.datetime.fromtimestamp(t) for t in result_df.t]
    result_df.index = [pd.Timestamp(datetime) for datetime in datetimes]
    result_df.loc[:, 'px_median'] = result_df['px'].rolling(
        '{}s'.format(mean_window), min_periods=1, center=False).median()
    result_df.loc[:, 'py_median'] = result_df['py'].rolling(
        '{}s'.format(mean_window), min_periods=1, center=False).median()
    return result_df


def filter_D_based_on_ground_truth(traj, ground_truth_pos, D):
    """  NOT USED FOR NOW.

    If the dataset is in-model, We can try to translate the times to "trajectory space"
    """
    from evaluate_dataset import get_length

    lengths = get_length(ground_truth_pos)
    lengths[np.isnan(lengths)] = 0  # because beginning of lengths can still have nans.
    assert len(lengths) == D.shape[0], len(lengths)

    # Use only distances for which we have valid ground truth.
    mask = list(lengths > 0)  # keep first zero length but delete others.
    mask[0] = True
    print('original D', D.shape)
    D = D[mask, :]
    print('reduced D to', D.shape)

    times = np.array(times)[mask]
    lengths = lengths[mask]

    assert len(times) == D.shape[0], len(times)

    time_diffs = times[1:] - times[:-1]
    velocities = lengths[1:] / time_diffs
    plt.figure()
    plt.hist(velocities, bins=20)
    plt.title('velocity histogram')

    distances = np.cumsum(lengths)
    times, *_ = traj.get_times_from_distances(arbitrary_distances=distances, time_steps=10000)
    return D, times


def plot_distance_errors(this_df, ax=None, **kwargs):
    indices = np.argsort(this_df.distance.values)
    distances = this_df.distance.values[indices]
    distances_gt = this_df.distance_gt.values[indices]
    errors = distances - distances_gt

    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(5, 2)
    ax.scatter(distances_gt, errors, alpha=0.5, **kwargs)
    ax.set_xlabel('real distance [m]')
    ax.set_ylabel('error [m]')
    ax.set_title(dataname)
    return ax


def plot_distance_times(this_df):
    range_ids = full_df[full_df.system_id == range_system_id].anchor_id.unique()
    fig, axs = plt.subplots(len(range_ids), sharex=True)
    fig.set_size_inches(10, 10)
    for i, anchor_id in enumerate(sorted(range_ids)):
        this_df = full_df[full_df.anchor_id == anchor_id]
        axs[i].scatter(this_df.timestamp, this_df.distance, color='red', label='measured distance')
        axs[i].scatter(this_df.timestamp, this_df.distance_gt, color='green', label='real distance')
        axs[i].legend(loc='upper right')
        axs[i].set_title('anchor {}'.format(anchor_id))
        axs[i].set_ylabel('distance [m]')
    axs[i].set_xlabel('time [s]')


def plot_individual(C_list, t_list, traj):
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 7)

    for Chat, t in zip(C_list, t_list):
        traj.set_coeffs(coeffs=Chat)
        if len(t) > 0:
            traj.plot(ax=ax, times=t)
            #traj.plot(ax=ax, times=t, label='{:.1f}'.format(t[0]))

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def plot_smooth(result_df):
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 7)
    #plt.scatter(result_df.px, result_df.py, s=1)
    plt.scatter(result_df.px_median, result_df.py_median, s=2, color='red')
    plt.plot(result_df.px_median, result_df.py_median, color='red')

    plt.plot(ground_truth_pos.px, ground_truth_pos.py, color='black')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


if __name__ == "__main__":
    anchor_names = None  # use all anchors by default.

    filename = 'datasets/uah1.mat'
    # fingers. works ok.
    #filename = 'datasets/Plaza1.mat'; # zig zag. does not work super well.
    #filename = 'datasets/Plaza2.mat' # triangle. works well.

    verbose = False
    traj = get_trajectory(filename)
    dataname = filename.split('/')[-1].split('.')[0]

    if dataname == 'uah1':
        t_window = 1.0
        eps = 2.0
        xlim = 0, 50
        ylim = -20, 20

        min_time = 0
        max_time = 1000

        # for iterative.
        n_complexity_it = 2
        model_it = 'polynomial'
        t_window_it = 80

    elif dataname == 'Plaza1':
        t_window = 0.5
        eps = 0.5
        xlim = -50, 10
        ylim = -20, 75

        # choose one:
        min_time = 0  # first big circle
        max_time = 200  # first big circle
        min_time = 510  # first loop
        max_time = 600  # first loop
        min_time = 0  # first few loops
        max_time = 1000  # first few loops.

        # for iterative.
        n_complexity_it = 3
        model_it = 'full_bandlimited'
        period_it = 40
        t_window_it = 20

    elif dataname == 'Plaza2':
        t_window = 0.1
        eps = 0.2
        xlim = -80, 10
        ylim = -20, 75

        min_time = 45.1
        period = 100.3 - 45.1
        num_loops = 2
        max_time = min_time + num_loops * period
        traj.period = period

        # for iterative.
        n_complexity_it = 5
        model_it = 'full_bandlimited'
        period_it = 40
        t_window_it = 40

    result_dict = loadmat(filename)

    ## Prepare dataset
    anchor_data = result_dict['TL']
    range_data = result_dict['TD']
    gt_data = result_dict['GT']

    anchors_df = create_anchors_df(anchor_data)
    anchors_df = format_anchors_df(anchors_df, range_system_id=range_system_id, gt_system_id=gt_system_id)

    full_df = create_full_df(range_data, gt_data)
    full_df = format_data_df(full_df, anchors_df, gt_system_id=gt_system_id, range_system_id=range_system_id)
    if verbose:
        print('time going from {:.1f} to {:.1f}'.format(full_df.timestamp.min(), full_df.timestamp.max()))
    full_df = full_df[(full_df.timestamp >= min_time) & (full_df.timestamp <= max_time)]
    full_df.loc[:, 'timestamp'] = full_df.timestamp - full_df.timestamp.min()

    fig, axs = plt.subplots(1, 2)
    sns.scatterplot(data=full_df, x='px', y='py', hue='timestamp', linewidth=0.0, ax=axs[0])
    sns.scatterplot(data=full_df, x='timestamp', y='px', hue='timestamp', linewidth=0.0, ax=axs[1])

    print('adding ground truth...')
    full_df = add_gt_raw(full_df, t_window=t_window, gt_system_id=gt_system_id)
    full_df.loc[:, "distance_gt"] = full_df.apply(
        lambda row: apply_distance_gt(row, anchors_df, gt_system_id=gt_system_id), axis=1)
    print('...done')

    fig, axs = plt.subplots(1, 2)
    range_df = full_df[full_df.system_id == range_system_id]
    sns.scatterplot(data=range_df, x='px', y='py', hue='timestamp', linewidth=0.0, ax=axs[0])
    #sns.scatterplot(data=anchors_df, x='px', y='py', linewidth=0.0,  ax=axs[0], color='red')
    sns.scatterplot(data=range_df, x='timestamp', y='px', hue='timestamp', linewidth=0.0, ax=axs[1])

    plot_distance_times(full_df)
    plot_distance_errors(full_df)

    anchors = get_coordinates(anchors_df, anchor_names)

    ## Construct anchors.
    if anchor_names is None:
        anchors = anchors_df.loc[:, ['px', 'py', 'pz']].values.astype(np.float32).T
    else:
        anchors_df = anchors_df.loc[anchors_df.anchor_name.isin(anchor_names)]
        anchors = get_coordinates(anchors_df, anchor_names)

    ## Construct times.
    range_df = full_df[full_df.system_id == range_system_id]
    times = range_df.timestamp.unique()

    ## Construct D.
    #chosen_distance = 'distance_gt'
    chosen_distance = 'distance'
    D, times = compute_distance_matrix(full_df, anchors_df, anchor_names, times, chosen_distance)
    if np.sum(D > 0) > D.shape[0]:
        print('Warning: multiple measurements for some times!')

    ## Construct ground truth.
    ground_truth_pos = get_ground_truth(full_df, times)

    list_complexities = [3, 5, 21, 51]
    for n_complexity in list_complexities:

        traj.set_n_complexity(n_complexity)
        basis = traj.get_basis(times=times)

        Chat_weighted = alternativePseudoInverse(D, anchors[:2, :], basis, weighted=True)
        Chat = alternativePseudoInverse(D, anchors[:2, :], basis, weighted=False)

        traj.set_coeffs(coeffs=Chat)

        traj_weighted = traj.copy()
        traj_weighted.set_coeffs(coeffs=Chat_weighted)

        fig, ax = plt.subplots()
        traj.plot(times=times, color='green', label='non-weighted', ax=ax)
        traj_weighted.plot(times=times, color='blue', label='weighted', ax=ax)
        ax.plot(full_df.px, full_df.py, color='black', label='ground truth')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title('K={}'.format(traj.n_complexity))
        ax.legend()

    ## Iterative algorithms
    traj.set_n_complexity(n_complexity_it)
    traj.model = model_it
    traj.period = period_it
    basis = traj.get_basis(times=times)

    ### Averaging algorithm
    C_list, t_list = averaging_algorithm(D, anchors[:2, :], basis, times, t_window=t_window_it)
    plot_individual(C_list, t_list, traj.copy())
    result_df = get_smooth_points(C_list, t_list, traj)
    plot_smooth(result_df)

    ### Build up algorithm
    C_list, t_list = build_up_algorithm(D, anchors[:2, :], basis, times, eps=eps, verbose=False)
    plot_individual(C_list, t_list, traj.copy())
    result_df = get_smooth_points(C_list, t_list, traj)
    plot_smooth(result_df)
