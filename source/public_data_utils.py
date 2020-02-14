# -*- coding: utf-8 -*-
"""
public_data_utils.py: Functions related specifically to the public datasets.

The goal of these functions is to create generic pandas dataframes that can be further processed
using functions in evaluate_dataset.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

from evaluate_dataset import format_anchors_df, format_data_df
from evaluate_dataset import add_gt_raw, apply_distance_gt
from trajectory_creator import get_trajectory

# Need to give different systems a name.
gt_system_id = "GT"
range_system_id = "Range"
gt_anchor_id = "GT"

# time intervals of zig zag trajectory in which movement is roughly linear.
TIME_RANGES = [
    (325, 350),  # backward
    (375, 393),  # forward
    (412, 445),
    (464, 484),
    (505, 534),
    (557, 575),
    (597, 624),
    (640, 670),
    (840, 867),
    (885, 908),
    (928, 961),
    (981, 1003),
    (1027, 1057),
    (1075, 1095),
    (1120, 1140),
    (1160, 1180),
    (1200, 1230),
    (1250, 1270),
    (1290, 1322),
    (1342, 1358),
]


def read_dataset(filename, verbose=False):
    traj = get_trajectory(filename)

    dataname = filename.split('/')[-1].split('.')[0]
    t_window = 1.0
    min_time = 0
    max_time = 10000
    if dataname == 'uah1':
        t_window = 1.0
        min_time = 0
        max_time = 1000
    elif dataname == 'Plaza1':
        t_window = 0.1
        min_time = 0  #20 straight lines
        max_time = 1400  # 20 straight lines
        #min_time = 325  # first line
        #max_time = 350  # first line
        #min_time = 374  # second line
        #max_time = 395  # second line
    elif dataname == 'Plaza2':
        t_window = 0.1
        min_time = 45.1
        period = 101 - 45
        num_loops = 2
        max_time = min_time + num_loops * period
        traj.period = period
    elif dataname == 'Gesling1':
        t_window = 2.0
        min_time = 36
        period = 140 - 36
        num_loops = 2
        max_time = min_time + num_loops * period
        traj.period = period
    elif dataname == 'Gesling2':
        t_window = 2.0
        min_time = 23
        period = 186 - 23
        num_loops = 1
        max_time = min_time + num_loops * period
        traj.period = period
    elif dataname == 'Gesling3':
        t_window = 1
        min_time = 23
        period = 50
        num_loops = 1
        max_time = min_time + num_loops * period
        if not traj.params['full_period']:
            traj.period = 2 * period

    try:
        result_dict = loadmat(filename)
    except FileNotFoundError:
        raise FileNotFoundError('Could not find {}. Did you run the script download_datasets?'.format(dataset))
    except Exception as e:
        print('Unknown reading error with {}. Check if the file looks ok.'.format(filename))
        raise e
    print('Successfully read {}'.format(filename))

    full_df, anchors_df = prepare_dataset(result_dict,
                                          range_system_id,
                                          gt_system_id, [min_time, max_time],
                                          t_window,
                                          verbose=verbose)
    return full_df, anchors_df, traj


def get_plotting_params(filename):
    xlim = ylim = (None, None)
    dataname = filename.split('/')[-1].split('.')[0]
    if dataname == 'uah1':
        xlim = 0, 50
        ylim = -20, 20
    elif dataname == 'Plaza1':
        xlim = -50, 10
        ylim = -20, 75
    elif dataname == 'Plaza2':
        xlim = -80, 10
        ylim = -20, 75
    elif 'Gesling' in dataname:
        xlim = -2, 50
        ylim = -2, 120
    return xlim, ylim


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


def create_full_df(range_data, gt_data, time_range=None):
    """" Create full dataframe. """
    mask = np.ones(len(range_data), dtype=bool)
    if time_range is not None:
        times = range_data[:, 0]
        times -= min(times)
        mask = (times > time_range[0]) & (times < time_range[1])
        if not any(mask):
            print('empty mask!')
            print(min(times), max(times), time_range)
    range_df = pd.DataFrame(columns=['timestamp', 'px', 'py', 'pz', 'distance', 'system_id', 'anchor_id'],
                            index=range(np.sum(mask)))
    range_df.loc[:, 'distance'] = range_data[mask, 3]
    range_df.loc[:, 'timestamp'] = range_data[mask, 0]
    range_df.loc[:, 'anchor_id'] = range_data[mask, 2]
    range_df.loc[:, 'system_id'] = range_system_id

    mask = np.ones(len(gt_data), dtype=bool)
    if time_range is not None:
        times = gt_data[:, 0]
        times -= min(times)
        mask = (times > time_range[0]) & (times < time_range[1])
    gt_df = pd.DataFrame(columns=range_df.columns)
    gt_df.loc[:, 'px'] = gt_data[mask, 1]
    gt_df.loc[:, 'py'] = gt_data[mask, 2]
    gt_df.loc[:, 'timestamp'] = gt_data[mask, 0]
    gt_df.loc[:, 'anchor_id'] = gt_anchor_id
    gt_df.loc[:, 'system_id'] = gt_system_id

    full_df = pd.concat([range_df, gt_df], ignore_index=True)
    full_df.sort_values('timestamp', inplace=True)
    full_df.reset_index(drop=True, inplace=True)
    full_df.loc[:, 'timestamp'] = full_df.timestamp - full_df.timestamp.min()
    return full_df


def prepare_dataset(result_dict, range_system_id, gt_system_id, time_range, t_window, verbose=False):
    min_time, max_time = time_range
    try:
        key_anchor = [key for key in result_dict.keys() if 'TL' in key][0]
        anchor_data = result_dict[key_anchor]
        key_range = [key for key in result_dict.keys() if 'TD' in key][0]
        range_data = result_dict[key_range]
        key_gt = [key for key in result_dict.keys() if 'GT' in key][0]
        gt_data = result_dict[key_gt]
    except KeyError:
        print('Problem reading')
        print(result_dict.keys())
        return

    anchors_df = create_anchors_df(anchor_data)
    anchors_df = format_anchors_df(anchors_df, range_system_id=range_system_id, gt_system_id=gt_system_id)

    if verbose:
        print('creating full_df...')
    full_df = create_full_df(range_data, gt_data, time_range)
    if len(full_df) == 0:
        raise ValueError('empty data frame')
    full_df = format_data_df(full_df, anchors_df, gt_system_id=gt_system_id, range_system_id=range_system_id)
    if verbose:
        print('...done')

    if verbose:
        print('adding ground truth...')
    #full_df = add_gt_raw(full_df, t_window=t_window, gt_system_id=gt_system_id)
    full_df.loc[:, ['px', 'py', 'pz']] = full_df.loc[:, ['px', 'py', 'pz']].fillna(method='ffill', limit=2)
    full_df.loc[:, "distance_gt"] = full_df.apply(
        lambda row: apply_distance_gt(row, anchors_df, gt_system_id=gt_system_id), axis=1)

    if verbose:
        print('...done')
    return full_df, anchors_df


def get_ground_truth(full_df, times):
    """ Find one ground truth for each time when we have a distance measurement. 
    """
    ground_truth_pos = full_df.loc[full_df.timestamp.isin(times), ['timestamp', 'px', 'py', 'pz']]
    ground_truth_pos = ground_truth_pos.astype(np.float32)
    ground_truth_pos = ground_truth_pos.groupby('timestamp').agg(np.nanmean)
    ground_truth_pos.reset_index(inplace=True)
    return ground_truth_pos.loc[:, ['px', 'py']]


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
