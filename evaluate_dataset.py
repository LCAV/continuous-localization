#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_dataset.py: Functions and pipeline to evaluate one dataset. 

See notebook RealExperiments for analysis for analysis and plotting of the 
intermediate results. 

"""

import numpy as np
import pandas as pd


def get_position(df, label="anchor_id", value=0):
    return df.loc[df[label] == value, ["px", "py", "pz"]].values.astype(np.float32)


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
    for anchor_id in new_df.anchor_id.unique():
        anchor_coord = get_position(anchors_df, "anchor_id", anchor_id)

        distances = np.linalg.norm(ground_truths[:, 1:] - anchor_coord, axis=1)

        # make sure that the time ordering is correct.
        timestamps = new_df.loc[new_df.anchor_id == anchor_id, "timestamp"].values.astype(np.float32)
        assert np.allclose(timestamps, ground_truths[:, 0])

        new_df.loc[new_df.anchor_id == anchor_id, label] = distances
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


def add_gt_raw(df, t_window=0.1, gt_system_id="Tango"):
    """ add median over t_window at each measurement point. 

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
