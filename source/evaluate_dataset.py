# -*- coding: utf-8 -*-
"""
evaluate_dataset.py: Functions and pipeline to evaluate datasets.

See notebook PublicDatasets for analysis and plotting of the 
intermediate results. 

"""
import datetime

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import scipy as sp

# These system_ids are used by the python measurement pipeline,
# but will be changed to better-readable "GT" and "Range", respectively.
from global_variables import TANGO_SYSTEM_ID, RTT_SYSTEM_ID

### IO


def read_correct_dataset(datafile, anchors_df, use_raw=False):
    if use_raw:
        data_df = read_dataset(datafile, anchors_df)
        print('reading', datafile)
        data_df = add_gt_raw(data_df, t_window=0.1)
        data_df.loc[:, 'distance_tango'] = data_df.apply(lambda row: apply_distance_gt(row, anchors_df), axis=1)
    else:
        datafile_root = datafile.split('.')[0]
        resample_name = datafile_root + '_resampled.pkl'
        print('reading', resample_name)
        data_df = pd.read_pickle(resample_name)
        change_system_ids(data_df)
        data_df = add_gt_resampled(data_df, anchors_df)
        data_df = format_data_df(data_df, anchors_df)
    return data_df


def read_anchors_df(anchorsfile):
    """ Read and preprocess the anchors file. """
    anchors_df = pd.read_csv(anchorsfile, engine='python')
    return format_anchors_df(anchors_df)


def read_dataset(datafile, anchors_df, gt_system_id=TANGO_SYSTEM_ID, range_system_id=RTT_SYSTEM_ID):
    """ Read and preprocess the measurement dataset, a .csv file. """
    data_df = pd.read_csv(datafile, engine='python')
    data_df.loc[:, 'timestamp'] = (data_df.timestamp.values - data_df.timestamp.min()) / 1000.  # in seconds
    return format_data_df(data_df, anchors_df, gt_system_id, range_system_id)


def format_anchors_df(anchors_df, gt_system_id=TANGO_SYSTEM_ID, range_system_id=RTT_SYSTEM_ID):
    """
    Make sure anchors_df is correctly formatted.
    """
    def apply_add_name(row, counter_dict):
        counter_dict[row.system_id] += 1
        return '{} {}'.format(row.system_id, counter_dict[row.system_id])

    anchors_df = anchors_df.astype({'anchor_id': str})
    anchors_df.loc[:, 'anchor_id'] = anchors_df.apply(lambda row: apply_anchor_id(row), axis=1)
    anchors_df.loc[:, 'system_id'] = anchors_df.apply(
        lambda row: apply_system_id(row, gt_system_id, range_system_id=range_system_id), axis=1)

    counter_dict = {name: -1 for name in anchors_df.system_id.unique()}
    anchors_df.loc[:, 'anchor_name'] = anchors_df.apply(lambda row: apply_add_name(row, counter_dict), axis=1)
    return anchors_df


def format_data_df(data_df, anchors_df=None, gt_system_id=TANGO_SYSTEM_ID, range_system_id=RTT_SYSTEM_ID):
    """
    Make sure data data_df is correctly formatted.
    """
    def filter_columns(data_df):
        all_columns = set(data_df.columns)
        keep_columns = set(
            ['timestamp', 'system_id', 'anchor_id', 'px', 'py', 'pz', 'distance', 'rssi', 'distance_gt', 'seconds'])
        drop_columns = all_columns - keep_columns  # this is a set difference
        data_df.drop(drop_columns, axis=1, inplace=True)

    filter_columns(data_df)
    data_df = data_df.astype({"anchor_id": str})
    data_df.loc[:, 'system_id'] = data_df.apply(lambda row: apply_system_id(row, gt_system_id, range_system_id), axis=1)
    if anchors_df is not None:
        data_df.loc[:, "anchor_name"] = data_df.apply(lambda row: apply_name(row, anchors_df), axis=1)
    return data_df


def change_system_ids(data_df):
    """ Temporary fix while results are still saved with all system ids. """
    if 'Tango' in data_df.system_id.unique():
        data_df.loc[data_df.system_id == 'Tango', 'system_id'] = 'GT'
    if 'RTT' in data_df.system_id.unique():
        data_df.loc[data_df.system_id == 'RTT', 'system_id'] = 'Range'


def apply_anchor_id(row):
    return row.anchor_id.strip()


def apply_system_id(row, gt_system_id=TANGO_SYSTEM_ID, range_system_id=RTT_SYSTEM_ID):
    if row.system_id == range_system_id:
        return 'Range'
    elif row.system_id == gt_system_id:
        return 'GT'
    else:
        return row.system_id  # do not change.


#### Geometry.


def convert_room_to_robot(x_room):
    """ Convert coordinates in room reference to robot reference. 
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


def match_reference(reference, points):
    """ Rotate and shift points to mach reference positions as closely as possible.
    Note that the order of points matters, not only their position.

    :param reference: 2D array of size (dimension, number of points), that does not change
    :param points: 2D array of points to rotate, the same size as reference
    :return:
        a pair (rotated points, (rotation matrix, rotation center, reference center of mass)))
    """
    assert reference.shape == points.shape
    reference_center = np.mean(reference, axis=1)
    reference -= reference_center[:, None]
    rotation_center = np.mean(points, axis=1)
    points -= rotation_center[:, None]
    rotation, e = sp.linalg.orthogonal_procrustes(points.T, reference.T)
    points = rotation @ points
    points += reference_center[:, None]
    return points, (rotation, rotation_center, reference_center)


#### Dataset processing.


def calibrate(original_df, gt_anchor_id='GT'):
    """ Calibrate for offset and slope. """
    assert 'distance_gt' in original_df.columns
    assert 'distance' in original_df.columns
    for anchor_id, anchor_df in original_df.groupby('anchor_id'):
        if anchor_id == gt_anchor_id:
            continue
        d_gt = anchor_df.distance_gt.values.astype(np.float32)
        d = anchor_df.distance.values.astype(np.float32)
        slope, offset = np.polyfit(x=d[~np.isnan(d)], y=d_gt[~np.isnan(d)], deg=1)
        original_df.loc[original_df.anchor_id == anchor_id, 'distance_calib'] = d * slope + offset
    print('added distance_calib column.')


def compute_distance_matrix(data_df,
                            anchors_df,
                            anchor_names=None,
                            times=None,
                            chosen_distance='distance',
                            dimension=3,
                            robot_height=0):
    """
    :param data_df: dataset which has time, distance, anchor_id data.
    :param anchors_df: dataset of anchors data. 
    :param anchor_names: list of anchor names to use. Set to None to use all.
    :param times: the measurement times which we want to use. Set to None to use all.
    :param chosen_distance: name of distance column to use.
    :param dimension: calculate distances in this dimension (2 or 3)
    :param robot_height: if dimension is 2, use this for robot height.

    :return: squared distance matrix of shape n_measurements x n_anchors.
    """

    if anchor_names is None:
        anchor_names = list(anchors_df.anchor_name.unique())
    if times is None:
        times = list(data_df.timestamp.unique())

    n_times = len(times)
    n_anchors = len(anchor_names)

    D_topright_real = np.zeros((n_times, n_anchors))

    i = 0
    actually_used_times = []
    for t in times:
        this_slice = data_df[(data_df.anchor_name.isin(anchor_names)) & (data_df.timestamp == t)]

        if len(this_slice) == 0:
            continue

        # this can be done more elegantly with pandas
        for anchor_name in this_slice.anchor_name:
            a_id = anchor_names.index(anchor_name)
            distance = this_slice.loc[this_slice.anchor_name == anchor_name, chosen_distance].values[0]
            if dimension == 3:
                D_topright_real[i, a_id] = distance**2
            else:
                if chosen_distance != 'distance_tango_2D':  # we already did this correction.
                    if not (anchor_name in anchors_df.anchor_name.unique()):
                        raise ValueError('{} not in {}'.format(anchor_name, anchors_df.anchor_name.unique()))
                    anchor_height = anchors_df[anchors_df.anchor_name == anchor_name].pz
                    distance_sq = distance**2 - (anchor_height - robot_height)**2
                else:
                    distance_sq = distance**2
                D_topright_real[i, a_id] = distance_sq

        actually_used_times.append(t)
        i += 1
    # If some times did not have valid measurements (not correct anchors, etc.)
    # then there might be some trailing all-zero rows.

    D_topright_real[np.isnan(D_topright_real)] = 0.0
    return D_topright_real[:i, :], actually_used_times


def compute_anchors(anchors_df, anchor_names=None):
    """ Sort anchors according to names, and return coordinates"""
    if anchor_names is not None:
        anchors_df = anchors_df.set_index('anchor_name')
        anchors_df = anchors_df.loc[anchor_names]
        anchors_df.reset_index(drop=False, inplace=True)
    all_ax = ['px', 'py', 'pz']
    for ax in all_ax:
        if any(np.isnan(anchors_df.loc[:, ax].values.astype(np.float32))):
            all_ax.remove(ax)
    anchors = anchors_df.loc[:, all_ax].values.astype(np.float32).T
    return anchors


def resample(data_df, t_range=[0, 100], t_delta=0.5, t_window=1.0, system_id="Range"):
    """ Resample measurements at regular timestamps. 

    :param data_df: dataframe with measurements. 
    :param t_range: tuple of min and max time, in seconds.
    :param t_delta: sampling interval, in seconds.
    :param t_window: window width used for median calculation, in seconds.
    :param system_id: which system to resample.
    """
    uniform_times = np.arange(*t_range, t_delta)

    if system_id == 'Range':
        fields = ["distance", "rssi"]
    elif system_id == 'GT':
        fields = ["px", "py", "pz"]

    anchor_ids = data_df[data_df.system_id == system_id].anchor_id.unique()

    len_new_df = len(uniform_times) * len(anchor_ids)
    new_df = pd.DataFrame(index=range(len_new_df), columns=data_df.columns)

    # distance measurements.
    i = 0
    for anchor_id in anchor_ids:
        df_anchor = data_df[data_df.anchor_id == anchor_id]
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


def add_gt_resampled(data_df, anchors_df, gt_system_id="GT", label='distance_gt'):
    """ Add ground truth distances to data_df as a new column.

    It uses the fact that the dataset is resampled, so we have perfectly synchronized measurements. 
    Therefore it takes less than 0.08 seconds on 4000 rows!! 
    """
    assert gt_system_id in data_df.system_id.values, '{} not in {}'.format(gt_system_id, data_df.system_id.unique())
    ground_truths = data_df.loc[data_df.system_id == gt_system_id, ["timestamp", "px", "py", "pz"]].values.astype(
        np.float32)

    for i, row in anchors_df.iterrows():
        if row.system_id == gt_system_id:
            continue
        anchor_coord = np.array([row.px, row.py, row.pz]).reshape((1, 3))
        distances = np.linalg.norm(ground_truths[:, 1:] - anchor_coord, axis=1)

        # calculate the tango distances in the plane.
        vecs = ground_truths[:, 1:3] - anchor_coord[0, :2]
        assert vecs.shape[0] == ground_truths.shape[0], vecs.shape
        assert vecs.shape[1] == 2, vecs.shape
        distances_2D = np.linalg.norm(vecs, axis=1)

        # make sure that the time ordering is correct.
        timestamps = data_df.loc[data_df.anchor_id == row.anchor_id, "timestamp"].values.astype(np.float32)
        assert np.allclose(timestamps, ground_truths[:, 0])

        data_df.loc[data_df.anchor_id == row.anchor_id, label] = distances
        data_df.loc[data_df.anchor_id == row.anchor_id, label + '_2D'] = distances_2D
    return data_df


def add_median_raw(data_df, t_window=1.0, range_system_id='Range'):
    """ Add (centered) median over t_window at each measurement point. 

    :param data_df: dataframe with measurements. 
    :param t_window: window width used for median calculation, in seconds.

    """
    for anchor_id, anchor_df in data_df[data_df.system_id == 'Range'].groupby("anchor_id"):
        print('processing', anchor_id)
        for t in anchor_df.timestamp:
            # we want to take into account all measurements that lie within the specified window.
            allowed = anchor_df.loc[np.abs(anchor_df.timestamp - t) <= t_window, "distance"]
            data_df.loc[(data_df.timestamp == t) &
                        (data_df.anchor_id == anchor_id), "distance_median"] = allowed.median()
            data_df.loc[(data_df.timestamp == t) & (data_df.anchor_id == anchor_id), "distance_mean"] = allowed.mean()
    return data_df


def add_median_raw_rolling(data_df, t_window=1000, gt_system_id="GT"):
    """ Add (non-cenetered) rolling median over t_window at each measurement point. 
    
    IMPORTANT: this is not centered. Our own implementation add_median_raw is centered. 
    However ours is much much slower...

    """
    data_df.sort_values("timestamp", inplace=True)
    datetimes = [datetime.datetime.fromtimestamp(t / 1000.0) for t in data_df.timestamp]
    data_df.index = [pd.Timestamp(datetime) for datetime in datetimes]
    for anchor_id, anchor_df in data_df.groupby('anchor_id'):
        system_id = anchor_df['system_id'].unique()[0]
        if system_id == gt_system_id:
            continue
        print('processing', anchor_id)
        rolling_data = anchor_df['distance'].rolling('{}ms'.format(t_window), min_periods=1, center=False)
        data_df.loc[data_df.anchor_id == anchor_id, "distance_mean"] = rolling_data.mean()
        data_df.loc[data_df.anchor_id == anchor_id, "distance_median"] = rolling_data.median()
    data_df.index = range(len(data_df))
    return data_df


def add_gt_raw(data_df, t_window=0.1, gt_system_id="GT"):
    """ Add median over t_window of ground truth position at each measurement point (centered).

    :param data_df: dataframe with measurements. 
    :param t_window: double window width used for median calculation, in seconds.

    """
    assert (gt_system_id in data_df.system_id.values), 'did not find any gt measurements in dataset.'
    df_gt = data_df[data_df.system_id == gt_system_id]

    coords = ['px', 'py', 'pz']
    if all(pd.isnull(data_df.pz)):
        coords = ['px', 'py']

    for i, row in data_df.iterrows():
        if row.system_id == gt_system_id:
            continue

        allowed = df_gt.loc[np.abs(df_gt.timestamp - row.timestamp) <= t_window, coords].astype(np.float32).values
        data_df.loc[i, coords] = np.nanmedian(allowed, axis=0)
    return data_df


def apply_name(row, anchors_df):
    if not any(anchors_df.anchor_id.isin([row.anchor_id])) and (row.anchor_id != 'GT'):
        print('Warning: {} not in {}'.format(row.anchor_id, anchors_df.anchor_id.unique()))
        return "unknown"
    elif row.anchor_id == 'GT':
        return 'GT'
    return anchors_df.loc[anchors_df.anchor_id.isin([row.anchor_id]), "anchor_name"].values[0]


def apply_distance_gt(row, anchors_df, gt_system_id="GT"):
    """ Return the ground truth distance between the measured anchor and the current ground truth. """
    if row.system_id == gt_system_id:
        return 0.0
    anchor_coord = anchors_df.loc[anchors_df.anchor_id == row.anchor_id, ['px', 'py', 'pz']].values.astype(
        np.float32).flatten()
    if len(anchor_coord) == 0:
        return np.nan
    point_coord = np.array([row.px, row.py, row.pz], dtype=np.float32)
    if np.isnan(anchor_coord[2]) or np.isnan(point_coord[2]):
        return np.linalg.norm(point_coord[:2] - anchor_coord[:2])
    else:
        return np.linalg.norm(point_coord - anchor_coord)


def apply_calibrate(row, calib_dict, calib_type):
    if row.system_id == 'GT':
        return 0.0
    offset = calib_dict[row.anchor_name][calib_type]
    return row.distance - offset


#### Detect start end end times.


def get_length(pos_df, plot=False):
    def diff(rolling):
        if len(rolling) > 1:
            return rolling[1] - rolling[0]
        else:
            return 0.0

    # To turn off annoying pandas warning. I did not figure out where it came from.
    pd.options.mode.chained_assignment = None

    # create new index so that we can use rolling function.
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


def find_start_times(tango_df, thresh_filter=-0.5, pattern=[1, 1, 1, 1, -1, -1], plot=False):
    """ Find the times at which the trajectory started. Can be multiple in one dataset.

    We find start times by fitting an "edge-detection" filter to the function of recorded traveled lengths over time.
    This way we (hopefully) reduce false starts triggered by spurious measurements.  

    :param tango_df: dataset with position estimates.
    :param thresh_filter: float, tuning parameter. if the response of the find_edge filter at a time is below this, it is considered a start.
    :param pattern: the pattern used in the find_edge function. 

    :return: two lists:
        - start_times: times at which movement starts.
        - start_indices:  indices at which movement starts.

    """
    def find_edge(window, pattern):
        """ Calculate the inner product of the pattern with the given window. """
        if len(window) < len(pattern):
            pattern = pattern[:-len(window)]
        sum_ = np.sum([a * b for (a, b) in zip(pattern, window)])
        return sum_

    if "length" not in tango_df.columns:
        tango_df.loc[:, "length"] = get_length(tango_df)

    # compute the rolling inner product between pattern and the length in tango_df.
    tango_df.index = [datetime.datetime.fromtimestamp(t) for t in tango_df.timestamp]
    normalized_series = tango_df.loc[:, "length"] / tango_df.length.max()
    df = normalized_series.rolling(window=len(pattern), center=False, min_periods=1).apply(find_edge,
                                                                                           raw=True,
                                                                                           kwargs={'pattern': pattern})
    tango_df.reset_index(inplace=True, drop=True)

    if plot:
        plt.figure()
        plt.plot(df.values, label='filter output')
        plt.plot([thresh_filter] * len(df), label='thresh_filterold')

    # find start indices
    all_indices = np.where(df.values < thresh_filter)[0]
    all_times = [tango_df.timestamp[i] for i in all_indices]

    # there were no calibration periods in this dataset.
    if len(all_indices) == 0:
        return [], []

    start_times = [all_times[0]]
    start_indices = [all_indices[0]]
    for s, i in zip(all_times, all_indices):
        if s > start_times[-1] + 10:  # assume that start indices cannot be closer than 10 seconds.
            start_times.append(s)
            start_indices.append(i)
    return start_times, start_indices


def find_end_times(tango_df, plot=False):
    """ Find the tines at which movement ends. """
    # TODO There is a shift of two here but it doesn't matter for now.
    end_times, end_indices = find_start_times(tango_df, thresh_filter=-0.2, pattern=[-1, -1, 1, 1, 1, 1], plot=plot)
    return end_times, end_indices


def find_calibration_data(tango_df, start_move_times, start_move_indices, max_length=0.01):
    """ Find calibration and trajectory times.
    
    :param tango_df: dataset with position estimates.
    :param start_move_times, start_move_indices: output of find_start_times. 
    :param max_length: if the travelled length is above this threshold, we consider that we moved.

    :return: dictionary with lists of calibration and trajectory times
        - calibration_data = {
         'calibration': [[t1_start, t1_end], [t2_start, t2_end], ...]
         'trajectory': [[t1_start, t1_end], [t2_start, t2_end], ...]}
    """
    calibration_data = {'calibration': [], 'trajectory': []}

    # Find calibration data
    # valid indices are close-to-zero length indices before the start times.
    for start_move_time, start_move_index in zip(start_move_times, start_move_indices):
        num_min = 3  # min number of calibration samples
        num_max = start_move_index  # max number of calibration samples.
        num_indices = 0
        for num_indices in range(num_min, num_max):
            if tango_df.iloc[start_move_index - num_indices].length > max_length:
                break

        start_calib_index = start_move_index - num_indices
        start_calib_time = tango_df.iloc[start_calib_index].timestamp

        # make previous trajectory and at the next calibration start.
        if len(calibration_data['trajectory']) > 0:
            calibration_data['trajectory'][-1].append(start_calib_time)

        calibration_data['calibration'].append([start_calib_time, start_move_time])
        calibration_data['trajectory'].append([start_move_time])

    # make last trajectory go until the end of dataset.
    calibration_data['trajectory'][-1].append(np.inf)
    return calibration_data
