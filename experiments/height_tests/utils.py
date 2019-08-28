#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

parameters = {
    'h1.csv':
    dict(
        phone_height=0.23,
        ap_height=0.93,
        real_distances={
            'd1': {
                'time': [0, 0.30],
                'distance': 0.5
            },
            'd2': {
                'time': [0.4, 1.10],
                'distance': 1.0
            },
            'd3': {
                'time': [1.25, 2.00],
                'distance': 2.5
            },
            'd4': {
                'time': [2.20, 3.00],
                'distance': 5.0
            },
            'd5': {
                'time': [3.20, 4.00],
                'distance': 7.5
            },
        }),
    'h2.csv':
    dict(
        phone_height=0.23,
        ap_height=0.93,
        real_distances={
            'd1': {
                'time': [0, 1.00],
                'distance': 7.5
            },
            'd2': {
                'time': [1.2, 1.50],
                'distance': 6
            },
            'd3': {
                'time': [2.1, 2.40],
                'distance': 5
            },
            'd4': {
                'time': [3.1, 3.40],
                'distance': 4
            },
            'd5': {
                'time': [4.1, 4.50],
                'distance': 3
            },
            'd6': {
                'time': [5.1, 5.40],
                'distance': 2.5
            },
            'd7': {
                'time': [6.1, 6.40],
                'distance': 1
            },
        }),
    'h3.csv':
    dict(
        phone_height=0.23,
        ap_height=0.93,
        real_distances={
            'd1': {
                'time': [0.0, 0.30],
                'distance': 1
            },
            'd2': {
                'time': [0.45, 1.15],
                'distance': 2
            },
            'd3': {
                'time': [1.3, 2.00],
                'distance': 3
            },
            'd4': {
                'time': [2.15, 2.45],
                'distance': 4
            },
            'd5': {
                'time': [3.0, 3.30],
                'distance': 5
            },
            'd6': {
                'time': [3.5, 4.20],
                'distance': 6
            },
            'd7': {
                'time': [4.4, 5.10],
                'distance': 7.5
            },
        }),
    'g1.csv':
    dict(
        #fname='g1.csv'
        phone_height=0.23,
        ap_height=0.93,
        real_distances={
            'd1': {
                'time': [0.0, 0.30],
                'distance': 1
            },
            'd2': {
                'time': [0.45, 1.15],
                'distance': 2
            },
            'd3': {
                'time': [1.3, 2.00],
                'distance': 3
            },
            'd4': {
                'time': [2.20, 2.50],
                'distance': 4
            },
            'd5': {
                'time': [3.05, 3.35],
                'distance': 5
            },
            'd6': {
                'time': [3.55, 4.25],
                'distance': 6
            },
            'd7': {
                'time': [4.45, 5.15],
                'distance': 7
            },
        }),
    'g2.csv':
    dict(
        phone_height=0.5,
        ap_height=0.93,
        real_distances={
            'd1': {
                'time': [0.0, 0.30],
                'distance': 1
            },
            'd2': {
                'time': [0.45, 1.15],
                'distance': 2
            },
            'd3': {
                'time': [1.3, 2.00],
                'distance': 3
            },
            'd4': {
                'time': [2.20, 2.50],
                'distance': 4
            },
            'd5': {
                'time': [3.05, 3.35],
                'distance': 5
            },
            'd6': {
                'time': [3.55, 4.25],
                'distance': 6
            },
            'd7': {
                'time': [4.45, 5.15],
                'distance': 7
            },
        }),
    'g3.csv':
    dict(
        phone_height=1.2,
        ap_height=0.93,
        real_distances={
            'd1': {
                'time': [0.00, 0.30],
                'distance': 1
            },
            'd2': {
                'time': [0.35, 1.05],
                'distance': 2
            },
            'd3': {
                'time': [1.10, 1.40],
                'distance': 3
            },
            'd4': {
                'time': [1.45, 2.15],
                'distance': 4
            },
            'd5': {
                'time': [2.20, 2.50],
                'distance': 5
            },
            'd6': {
                'time': [2.55, 3.25],
                'distance': 6
            },
            'd7': {
                'time': [3.30, 4.00],
                'distance': 7
            },
        }),
    'g4.csv':
    dict(
        phone_height=1.2,
        ap_height=0.93,
        real_distances={
            'd1': {
                'time': [0.00, 0.30],
                'distance': 1
            },
            'd2': {
                'time': [0.35, 1.05],
                'distance': 2
            },
            'd3': {
                'time': [1.10, 1.40],
                'distance': 3
            },
            'd4': {
                'time': [1.45, 2.15],
                'distance': 4
            },
            'd5': {
                'time': [2.20, 2.50],
                'distance': 5
            },
            'd6': {
                'time': [2.55, 3.25],
                'distance': 6
            },
            'd7': {
                'time': [3.30, 4.00],
                'distance': 7
            },
        }),
    'g5.csv':
    dict(
        phone_height=1.2,
        ap_height=0.93,
        real_distances={
            'd1': {
                'time': [0.00, 0.30],
                'distance': 1
            },
            'd2': {
                'time': [0.35, 1.05],
                'distance': 2
            },
            'd3': {
                'time': [1.10, 1.40],
                'distance': 3
            },
            'd4': {
                'time': [1.45, 2.15],
                'distance': 4
            },
            'd5': {
                'time': [2.20, 2.50],
                'distance': 5
            },
            'd6': {
                'time': [2.55, 3.25],
                'distance': 6
            },
            'd7': {
                'time': [3.30, 4.00],
                'distance': 7
            },
        }),
    'f1.csv':
    dict(
        phone_height=0.5,
        ap_height=0.93,
        real_distances={
            'd1': {
                'time': [0.00, 0.30],
                'distance': 7
            },  # face west
            'd2': {
                'time': [0.50, 1.20],
                'distance': 7
            },  # face south
            'd3': {
                'time': [1.30, 2.10],
                'distance': 7
            },  # face east
            'd4': {
                'time': [2.30, 3.00],
                'distance': 7
            },  # face north
        }),
    'f2.csv':
    dict(
        phone_height=0.5,
        ap_height=0.93,
        real_distances={
            'd1': {
                'time': [0.00, 0.30],
                'distance': 5
            },  # face west
            'd2': {
                'time': [0.50, 1.20],
                'distance': 5
            },  # face south
            'd3': {
                'time': [1.30, 2.10],
                'distance': 5
            },  # face east
            'd4': {
                'time': [2.30, 3.00],
                'distance': 5
            },  # face north
        }),
    'f3.csv':
    dict(
        phone_height=0.5,
        ap_height=0.93,
        real_distances={
            'd1': {
                'time': [0.00, 0.30],
                'distance': 2
            },  # face west
            'd2': {
                'time': [0.50, 1.20],
                'distance': 2
            },  # face south
            'd3': {
                'time': [1.30, 2.10],
                'distance': 2
            },  # face east
            'd4': {
                'time': [2.30, 3.00],
                'distance': 2
            },  # face north
        }),
}


def convert_to_seconds(float_time, verbose=False):
    seconds = np.round(float_time % 1.0, 3)
    minutes = np.round(float_time - seconds, 0)
    if verbose:
        print(seconds, minutes)
    return minutes * 60 + seconds * 100


def read_and_plot(fname, params, verbose=False, ax=None, y='dist_var'):
    data = pd.read_csv(fname)
    data.timestamp /= 1000.
    data.timestamp -= data.timestamp[0]
    data_reduced = pd.DataFrame(columns=data.columns)
    for dict_ in params['real_distances'].values():
        tmin, tmax = dict_['time']
        tmin_sec = convert_to_seconds(tmin)
        tmax_sec = convert_to_seconds(tmax, verbose=verbose)
        dreal = (dict_['distance']**2 + (params['ap_height'] - params['phone_height'])**2)**0.5
        mask = (data.timestamp <= tmax_sec) & (data.timestamp >= tmin_sec)
        data.loc[mask, 'distance_real'] = dreal
        data.loc[mask, 'distance_median'] = data[mask].distance.median()
        data.loc[mask, 'distance_min'] = data[mask].distance.min()
        data_reduced = pd.concat([data_reduced, data[mask]], ignore_index=True, sort=True)

    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(data_reduced.timestamp, data_reduced.distance, label='measured')
    ax.scatter(data_reduced.timestamp, data_reduced.distance_real, label='real')
    #ax.scatter(data_reduced.timestamp, data_reduced.distance_median, label='median')
    ax.scatter(data_reduced.timestamp, data_reduced.distance_min, label='min')
    ax.legend(loc='best')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('distance [m]')
    ax.set_title('{}, height={}'.format(fname, params['phone_height']))
    ax.set_ylim(-5, 20)

    if y == 'rssi':
        axy = ax.twinx()
        axy.scatter(data_reduced.timestamp, data_reduced.rssi, label='rssi', color='green', marker='+')
        axy.set_ylabel('rssi [dBm]', color='green')
        axy.tick_params(axis='y', labelcolor='green')
        if any(data_reduced.rssi.values > -30):
            print('warning', np.max(data_reduced.rssi.values))
        if any(data_reduced.rssi.values < -64):
            print('warning', np.min(data_reduced.rssi.values))
        axy.set_ylim(-64, -30)
    if y == 'dist_var':
        axy = ax.twinx()
        axy.scatter(data_reduced.timestamp, data_reduced.dist_var, label='dist_var', color='green', marker='+')
        axy.set_ylabel('dist var', color='green')
        axy.tick_params(axis='y', labelcolor='green')
        #axy.set_ylim(-64, -30)


if __name__ == "__main__":
    assert convert_to_seconds(1.15) == 75
    assert convert_to_seconds(3.00) == 180
