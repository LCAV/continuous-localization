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


def create_anchors_df(anchor_data):
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
    gt_df.loc[:, 'px'] = gt_data[:, 1]
    gt_df.loc[:, 'py'] = gt_data[:, 2]
    gt_df.loc[:, 'timestamp'] = gt_data[:, 0]
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
    result_df.loc[:, 'px_median'] = result_df['px'].rolling('{}s'.format(mean_window), min_periods=1,
                                                            center=False).median()
    result_df.loc[:, 'py_median'] = result_df['py'].rolling('{}s'.format(mean_window), min_periods=1,
                                                            center=False).median()
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

    # a quick hack for calculating the variance.
    error_df = pd.DataFrame({'e': errors, 'd': distances_gt})
    error_df.sort_values('d', inplace=True)
    variances = error_df.e.rolling(10).std().values
    print('mean std', np.nanmean(variances))
    print('median std', np.nanmedian(variances))

    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(5, 2)
    ax.scatter(distances_gt, errors, alpha=0.5, **kwargs)
    ax.set_xlabel('real distance [m]')
    ax.set_ylabel('distance error [m]')
    return ax, errors, distances_gt


def plot_distance_times(full_df):
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
    return fig, axs


def plot_individual(C_list, t_list, traj):
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 7)

    for Chat, t in zip(C_list, t_list):
        traj.set_coeffs(coeffs=Chat)
        if len(t) > 0:
            traj.plot(ax=ax, times=t)
            #traj.plot(ax=ax, times=t, label='{:.1f}'.format(t[0]))
    return ax


def plot_smooth(result_df):
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 7)
    #plt.scatter(result_df.px, result_df.py, s=1)
    plt.scatter(result_df.px_median, result_df.py_median, s=2, color='red')
    plt.plot(result_df.px_median, result_df.py_median, color='red')
    return ax


if __name__ == "__main__":
    print('see Datasets.ipynb for how to use above functions.')
