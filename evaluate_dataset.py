#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_dataset.py: Functions and pipeline to evaluate one dataset. 

See notebook RealExperiments for analysis for analysis and plotting of the 
intermediate results. 

"""
import datetime

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

tango_system_id = 7585
rtt_system_id = 7592


def resample(df, t_range=[0, 100], t_delta=0.5, t_window=1.0, system_id="RTT"):
    """ Resample measurements at regular timestamps. 

    :param df: dataframe with measurements. 
    :param t_range: tuple of min and max time, in seconds.
    :param t_delta: sampling interval, in seconds.
    :param t_window: window width used for median calculation, in seconds.

    """
    uniform_times = np.arange(*t_range, t_delta)

    if system_id == 'RTT':
        fields = ["distance", "rssi"]
    elif system_id == 'Tango':
        fields = ["px", "py", "pz"]

    anchor_ids = df[df.system_id == system_id].anchor_id.unique()

    len_new_df = len(uniform_times) * len(anchor_ids)
    new_df = pd.DataFrame(index=range(len_new_df), columns=df.columns)

    # distance measurements.
    i = 0
    for anchor_id in anchor_ids:
        df_anchor = df[df.anchor_id == anchor_id]
        system_id = df_anchor.system_id.unique()[0]
        for t in uniform_times:
            if i % 100 == 0:
                print('{}/{}'.format(i, len_new_df))
            valid_data = df_anchor[np.abs(df_anchor.timestamp - t) <= (t_window / 0.2)]
            if len(valid_data) > 0:
                new_df.loc[i, fields] = valid_data.loc[:, fields].median().values
            else:
                print('Warning: no data for anchor {} at time {}'.format(anchor_id, t))
            new_df.loc[i, ['timestamp', 'anchor_id', 'system_id']] = t, anchor_id, system_id
            i += 1
    new_df.sort_values('timestamp', inplace=True)
    return new_df


def add_gt_resampled(new_df, anchors_df, gt_system_id="Tango", label='distance_tango'):
    """ This takes less than 0.08 seconds on 4000 rows!! 
    
    It uses the fact that the dataset is resampled, so we have perfectly synchronized measurements. 
    
    """
    ground_truths = new_df.loc[new_df.system_id == gt_system_id, ["timestamp", "px", "py", "pz"]].values.astype(
        np.float32)

    for i, row in anchors_df.iterrows():
        if row.system_id == gt_system_id:
            continue
        anchor_coord = np.array([row.px, row.py, row.pz]).reshape((1, 3))
        distances = np.linalg.norm(ground_truths[:, 1:] - anchor_coord, axis=1)

        # make sure that the time ordering is correct.
        timestamps = new_df.loc[new_df.anchor_id == row.anchor_id, "timestamp"].values.astype(np.float32)
        assert np.allclose(timestamps, ground_truths[:, 0])

        new_df.loc[new_df.anchor_id == row.anchor_id, label] = distances
    return new_df


def add_median_raw(df, t_window=1.0):
    """ Add median over t_window at each measurement point. 

    :param df: dataframe with measurements. 
    :param t_window: window width used for median calculation, in seconds.

    """
    for a_id in df[df.system_id == 'RTT'].anchor_id.unique():
        print('processing', a_id)
        anchor_df = df[df.anchor_id == a_id]
        for t in anchor_df.timestamp:
            # we want to take into account all measurements that lie within the specified window.
            allowed = anchor_df.loc[np.abs(anchor_df.timestamp - t) <= t_window, "distance"]
            df.loc[(df.timestamp == t) & (df.anchor_id == a_id), "distance_median"] = allowed.median()
            df.loc[(df.timestamp == t) & (df.anchor_id == a_id), "distance_mean"] = allowed.mean()
    return df


def add_median_raw_rolling(data_df):
    """ IMPORTANT: this is not centered. Our own implementation add_median_raw is centered. 
     However ours is much much slower...
     """
    data_df.sort_values("timestamp", inplace=True)
    datetimes = [datetime.datetime.fromtimestamp(t / 1000.0) for t in data_df.timestamp]
    data_df.index = [pd.Timestamp(datetime) for datetime in datetimes]
    for anchor_id, anchor_df in data_df.groupby('anchor_id'):
        rolling_data = anchor_df['distance'].rolling('1s', min_periods=1, center=False)
        data_df.loc[data_df.anchor_id == anchor_id, "distance_mean"] = rolling_data.mean()
        data_df.loc[data_df.anchor_id == anchor_id, "distance_median"] = rolling_data.median()
    data_df.index = range(len(data_df))


def add_gt_raw(df, t_window=0.1, gt_system_id="Tango"):
    """ Add median over t_window at each measurement point. 

    :param df: dataframe with measurements. 
    :param t_window: window width used for median calculation, in seconds.

    """
    assert (gt_system_id in df.system_id.values), 'did not find any gt measurements in dataset.'
    df_gt = df[df.system_id == 'Tango']

    for i, row in df.iterrows():
        if row.system_id == gt_system_id:
            continue
        elif row.system_id == 'RTT':
            allowed = df_gt.loc[np.abs(df_gt.timestamp - row.timestamp) <= t_window, ['px', 'py', 'pz']]
            df.loc[i, ['px', 'py', 'pz']] = allowed.median()
        else:
            raise ValueError(row.system_id)
    return df


def read_anchors_df(anchorsfile):
    def apply_add_name(row, rtt_, tango_):
        if row.system_id == 'Tango':
            tango_[0] += 1
            return 'Tango {}'.format(tango_[0])
        elif row.system_id == 'RTT':
            rtt_[0] += 1
            return 'RTT {}'.format(rtt_[0])

    def apply_system_id(row):
        if row.system_id == tango_system_id:
            return 'Tango'
        elif row.system_id == rtt_system_id:
            return 'RTT'

    def apply_anchor_id(row):
        return row.anchor_id.strip()

    rtt_ = [-1]
    tango_ = [-1]

    anchors_df = pd.read_csv(anchorsfile)
    anchors_df.loc[:, 'anchor_id'] = anchors_df.apply(lambda row: apply_anchor_id(row), axis=1)
    anchors_df.loc[:, 'system_id'] = anchors_df.apply(lambda row: apply_system_id(row), axis=1)
    anchors_df.loc[:, 'anchor_name'] = anchors_df.apply(lambda row: apply_add_name(row, rtt_, tango_), axis=1)
    return anchors_df


def apply_name(row, anchors_df):
    return anchors_df.loc[anchors_df.anchor_id == row.anchor_id, "anchor_name"].values[0]


def read_dataset(datafile, anchors_df):
    def apply_system_id(row):
        if row.system_id == rtt_system_id:
            return 'RTT'
        elif row.system_id == tango_system_id:
            return 'Tango'
        else:
            raise NameError(row.system_id)

    def filter_columns(data_df):
        all_columns = set(data_df.columns)
        keep_columns = set([
            'timestamp', 'system_id', 'anchor_id', 'px', 'py', 'pz', 'theta_x', 'theta_y', 'theta_z', 'distance',
            'rssi', 'seconds'
        ])
        drop_columns = all_columns - keep_columns  # this is a set difference
        data_df.drop(drop_columns, axis=1, inplace=True)

    data_df = pd.read_csv(datafile)
    filter_columns(data_df)
    data_df.loc[:, 'timestamp'] = (data_df.timestamp.values - data_df.timestamp.min()) / 1000.  # in seconds
    data_df.loc[:, 'anchor_name'] = data_df.apply(lambda row: apply_name(row, anchors_df), axis=1)
    data_df.loc[:, 'system_id'] = data_df.apply(lambda row: apply_system_id(row), axis=1)
    return data_df


def apply_distance_gt(row, anchors_df, gt_system_id="Tango"):
    if row.system_id == gt_system_id:
        return 0.0
    anchor_coord = anchors_df.loc[anchors_df.anchor_id == row.anchor_id, ['px', 'py', 'pz']].values.astype(np.float32)
    point_coord = np.array([row.px, row.py, row.pz], dtype=np.float32)
    return np.linalg.norm(point_coord - anchor_coord)


def apply_calibrate(row, calib_dict, calib_type):
    if row.system_id == 'Tango':
        return 0.0
    offset = calib_dict[row.anchor_name][calib_type]
    return row.distance - offset


def convert_room_to_robot(x_room):
    """
    Convert coordinates in room reference to robot reference. 
    """
    assert x_room.shape[0] == 3
    assert x_room.ndim > 1 and x_room.shape[1] >= 1

    from math import cos, sin, pi

    theta = pi / 2.0
    R = np.array([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]])
    #origin = np.array([3.4, 3.58, 0.0]).reshape((3, 1))
    origin = np.array([3.58, 3.4, 0.0]).reshape((3, 1))
    x_robot = R.dot(x_room - origin)
    return x_robot


#### Detect start end end times.


def get_length(pos_df, plot=False):
    def diff(rolling):
        if len(rolling) > 1:
            return rolling[1] - rolling[0]
        else:
            return 0.0

    # To turn off annoying pandas warning. I did not figure out where it came from.
    pd.options.mode.chained_assignment = None

    new_index = [datetime.datetime.fromtimestamp(t) for t in pos_df.timestamp]
    pos_df.index = new_index
    v_x = pos_df.px.rolling(min_periods=1, window=2, center=False).apply(func=diff, raw=True)
    v_y = pos_df.py.rolling(min_periods=1, window=2, center=False).apply(func=diff, raw=True)
    d_t = pos_df.timestamp.rolling(min_periods=1, window=2, center=False).apply(func=diff, raw=True)
    pos_df.reset_index(inplace=True, drop=True)

    v = np.vstack((v_x.values.astype(np.float32), v_y.values.astype(np.float32))).T
    lengths = np.linalg.norm(v, axis=1)

    if plot:
        # TODO this is numerically bad, velocities are extremely noisy...
        # That's why we are not using velocities but lengths.
        d_times = d_t.values.astype(np.float32).reshape((-1, ))
        velocities = lengths / d_times
        median = np.median(velocities[d_times >= np.median(d_times)])
        velocities[d_times < np.median(d_times)] = median
        fig, axs = plt.subplots(3, 1)
        fig.set_size_inches(15, 5)
        axs[0].plot(d_times, label='d_times')
        axs[1].plot(lengths, label='lengths')
        axs[2].plot(velocities, label='length')
        axs[2].set_ylim([0, 0.3])
    return lengths


def find_times(tango_df):
    """ This was an attempt at a more straight forward way to find calibraiton
    or movement times. After all, find_start_times worked better.

    """
    movement_times = []
    calibration_times = []

    threshold = 1e-4

    times = tango_df.timestamp

    i = 0
    l = tango_df.loc[i, "length"]
    moving = (l >= threshold)
    if moving:
        movement_times.append([i, -1])
    else:
        calibration_times.append([i, -1])

    for i in range(1, len(times)):
        l = tango_df.loc[i, "length"]

        if moving and (l >= threshold):
            # still moving.
            pass
        elif moving and (l <= threshold):
            # stopped moving.
            movement_times[-1][1] = i - 1
            calibration_times.append([i, -1])
            moving = False
        elif (not moving) and (l <= threshold):
            # still not moving.
            pass
        elif (not moving) and (l >= threshold):
            # started moving.
            calibration_times[-1][1] = i - 1
            movement_times.append([i, -1])
            moving = True
    return movement_times, calibration_times


def find_start_times(tango_df, thresh_filter=-0.5, pattern=[1, 1, 1, 1, -1, -1], plot=False):
    def find_edge(window, pattern=[1, 1, -1]):
        if len(window) < len(pattern):
            pattern = pattern[:-len(window)]
        sum_ = np.sum([a * b for (a, b) in zip(pattern, window)])
        return sum_

    if "length" not in tango_df.columns:
        tango_df.loc[:, "length"] = get_length(tango_df)

    tango_df.index = [datetime.datetime.fromtimestamp(t) for t in tango_df.timestamp]
    normalized_series = tango_df.loc[:, "length"] / tango_df.length.max()
    df = normalized_series.rolling(
        window=6, center=False, min_periods=1).apply(
            find_edge, raw=True, kwargs={'pattern': pattern})
    tango_df.reset_index(inplace=True, drop=True)

    if plot:
        plt.figure()
        plt.plot(df.values, label='filter output')
        #plt.plot([thresh_filter]*len(df), label='thresh_filterold')

    # find start indices
    all_indices = np.where(df.values < thresh_filter)[0]
    all_times = [tango_df.timestamp[i] for i in all_indices]

    start_times = [all_times[0]]
    start_indices = [all_indices[0]]
    for s, i in zip(all_times, all_indices):
        if s > start_times[-1] + 10:  # assume that start indices cannot be closer than 10 seconds.
            start_times.append(s)
            start_indices.append(i)
    return start_times, start_indices


def find_end_times(tango_df, plot=False):
    # TODO There is a shift of two here but it doesn't matter for now.
    end_times, end_indices = find_start_times(tango_df, thresh_filter=-0.2, pattern=[-1, -1, 1, 1, 1, 1], plot=plot)
    return end_times, end_indices


def find_calibration_data(tango_df, start_times, start_indices, max_length=0.01):
    # Find calibration data
    # valid indices are close-to-zero length indices before the start times.
    calibration_data = {}

    # weird name inconsistency because in terms of calibration start_time is the end-time.
    for end_time, end_index in zip(start_times, start_indices):
        num_min = 5
        num_max = min(100, end_index)
        for num_indices in range(num_min, num_max):
            if tango_df.iloc[end_index - num_indices].length > max_length:
                break

        start_index = end_index - num_indices
        start_time = tango_df.iloc[start_index].timestamp
        calibration_data[start_index] = [start_time, end_time]
    return calibration_data
