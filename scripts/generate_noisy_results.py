#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scripts to generate data for Fig. 6. from the Relax and Recover paper, see GenerateAllFigures.ipynb"""

import sys
from os.path import abspath, dirname
this_dir = dirname(abspath(__file__))
sys.path.append(this_dir + '/../source')

from multiprocessing import Pool

from simulation import run_simulation
from simulation import arg_parser


def f(params):
    if params["outfolder"] != '':
        outfolder = f'{params["outfolder"]}/{params["key"]}/'
    else:
        outfolder = None
    run_simulation(params, outfolder, solver=params['solver'], verbose=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outfolder', metavar='outfolder', help='output folder', type=str, default='')
    parser.add_argument('-n', '--n_it', metavar='n_it', type=int, help='number of iterations', default=20)
    args = parser.parse_args()

    #n_it = 1000
    #outfolder = this_dir + '/../results/
    outfolder = args.outfolder
    n_it = args.n_it

    noise_simgas = [1e-1, 1e0, 1e1]

    parameters = [
        {
            'key': 'noise_to_square_right_inverse',
            'n_its': n_it,
            'time': 'undefined',
            'positions': [500],
            'complexities': [5],
            'anchors': [4],
            'noise_sigmas': noise_simgas,
            'success_thresholds': [0.0] * len(noise_simgas),
            'noise_to_square': True,
            'solver': 'trajectory_recovery',
            'sampling_strategy': 'single_time'
        },
        {
            'key': 'noise_right_inverse_weighted',
            'n_its': n_it,
            'time': 'undefined',
            'positions': [500],
            'complexities': [5],
            'anchors': [4],
            'noise_sigmas': noise_simgas,
            'success_thresholds': [0.0] * len(noise_simgas),
            "noise_to_square": False,
            'solver': 'weighted_trajectory_recovery',
            'sampling_strategy': 'single_time'
        },
        {
            'key': 'noise_right_inverse',
            'n_its': n_it,
            'time': 'undefined',
            'positions': [500],
            'complexities': [5],
            'anchors': [4],
            'noise_sigmas': noise_simgas,
            'success_thresholds': [0.0] * len(noise_simgas),
            "noise_to_square": False,
            'solver': 'trajectory_recovery',
            'sampling_strategy': 'single_time'
        },
        {
            'key': 'noise_and_anchors',
            'n_its': n_it,
            'time': 'undefined',
            'positions': [500],
            'complexities': [3],
            'anchors': [3, 5, 10],
            'noise_sigmas': [1],
            'success_thresholds': [0.0],
            'noise_to_square': True,
            'solver': 'trajectory_recovery',
            'sampling_strategy': 'single_time'
        },
    ]
    for p in parameters:
        p['outfolder'] = outfolder

    with Pool(4) as p:
        p.map(f, parameters)
