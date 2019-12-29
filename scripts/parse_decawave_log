#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parser.py: Parse logfiles of Decawave to pandas DataFrames.
"""

import pandas as pd

counter = 0


def process_line(result_df, line):
    import re
    if not 'distances: ' in line:
        return

    x = y = z = q = None
    match = re.search('position: x=(\d+)', line)
    if match:
        x = float(match.group(1)) / 1000.
    match = re.search('y=(\d+)', line)
    if match:
        y = float(match.group(1)) / 1000.
    match = re.search('z=(\d+)', line)
    if match:
        z = float(match.group(1)) / 1000.
    match = re.search('q=(\d+)', line)
    if match:
        q = float(match.group(1))

    # TODO(FD) below could be done more elegantly using
    # the above regex functions.

    time = line.split(':')[0]
    device_id = line.split(' location data:')[0][-4:]

    data = line.split('distances: ')[-1]
    meas_list = data.split('}, ')
    for arr in meas_list:
        anchor_id, length, qual, *_ = arr.split(' ')
        try:
            d = float(length.strip('distance=Distance{length=').strip()[:-1]) / 1000.  # remove trailing comma.
            q = qual.strip('quality=').strip().strip('}')
        except Exception as e:
            print('error parsing', length.split('distance=Distance{length='))
            raise
        result_df.loc[len(result_df)] = dict(time=time,
                                             device_id=device_id,
                                             anchor_id=anchor_id,
                                             distance=d,
                                             quality=q,
                                             px=x,
                                             py=y,
                                             pz=z,
                                             pq=q)


def anchor_line(anchor_df, line):
    if not ('initial-position' in line):
        return False

    arr = line.split(' ')
    full_id = arr[1]
    anchor_id = full_id[-4:]
    x = arr[3].strip('x=')
    y = arr[4].strip('y=')
    z = arr[5].strip('z=')
    anchor_df.loc[len(anchor_df)] = dict(anchor_id=anchor_id, px=x, py=y, pz=z, full_id=full_id)
    return True


def parse_file(logfile):
    result_df = pd.DataFrame(columns=['time', 'device_id', 'anchor_id', 'distance', 'quality', 'px', 'py', 'pz', 'pq'])
    anchor_df = pd.DataFrame(columns=['anchor_id', 'px', 'py', 'pz', 'full_id'])
    with open(logfile, 'r') as f:
        counter = 0
        line = f.readline()
        while anchor_line(anchor_df, line):
            line = f.readline()
            pass

        print('anchor_df')
        print(anchor_df)

        counter = 0
        process_line(result_df, line)
        while line:
            line = f.readline()
            process_line(result_df, line)

    result_df = result_df.apply(pd.to_numeric, errors='ignore', axis=0)
    anchor_df = anchor_df.apply(pd.to_numeric, errors='ignore', axis=0)
    return result_df, anchor_df


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Parse decawave logfile.')
    parser.add_argument('filenames', metavar='fname', type=str, nargs='+', help='filenames to process')
    args = parser.parse_args()
    logfiles = args.filenames  #'UWB/monday.txt'
    for logfile in logfiles:
        print('processing', logfile)

        result_df, anchor_df = parse_file(logfile)
        print('result_df')
        print(result_df.head())

        resultfile = logfile.replace('.txt', '.pkl')
        result_df.to_pickle(resultfile)
        print('saved as', resultfile)

        resultfile = logfile.replace('.txt', '_anchors.pkl')
        if not anchor_df.empty:
            anchor_df.to_pickle(resultfile)
            print('saved as', resultfile)
