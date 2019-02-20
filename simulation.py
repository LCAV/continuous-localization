#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cvxpy
import os
import time
import logging

from trajectory import Trajectory
from environment import Environment
from global_variables import DIM, TMAX
from solvers import OPTIONS, semidefRelaxationNoiseless, rightInverseOfConstraints, alternativePseudoInverse
from json_io import *
"""
simulation.py: 
"""

def robust_increment(arr, idx):
    """ increment value of array if inside bound, and set to 1 if previously nan. """
    if idx < arr.shape:
        if np.isnan(arr[idx]):
            arr[idx] = 1
        else:
            arr[idx] += 1


def robust_add(arr, idx, value):
    if idx < arr.shape:
        if np.isnan(arr[idx]):
            arr[idx] = 0
        arr[idx] += value


def run_simulation(parameters, outfolder=None, solver=None):
    """ Run simulation. 

    :param parameters: Can be either the name of the folder where parameters.json is stored, or a new dict of parameters.
    """

    if outfolder is not None and not os.path.exists(outfolder):
        os.makedirs(outfolder)

    if type(parameters) == str:
        fname = parameters + 'parameters.json'
        p = read_json(fname)
        print('read parameters from file {}.'.format(fname))
        print('parameters:', p)
    elif type(parameters) == dict:
        # If we are trying new parameters and saving in a directory that already exists,
        # we need to make sure that the saved parameters are actually the same.
        p = parameters
        if outfolder is not None:
            try:
                
                parameters_old = read_json(outfolder + 'parameters.json')
                p['time'] = parameters_old['time']
                assert p == parameters_old, 'Found parameters file with different content than new parameters!'
            except FileNotFoundError:
                print('Did not find existing parameters file.')
            except AssertionError as error:
                raise (error)
    else:
        raise TypeError('parameters needs to be folder name or dictionary.')


    out_shape = (len(p['n_complexity']), len(p['n_anchors']), len(p['n_samples']), 
                 len(p['sigmas_noise']), len(p['values_missing']))
    results_header = ['n_complexity', 'n_anchors', 'n_samples', 
                       'sigmas_noise', 'values_missing']

    keys = ['successes', 'errors', 'num_not_solved']
    results = {k:np.full(out_shape, np.nan) for k in keys}

    for c_idx, n_complexity in enumerate(p['n_complexity']):
        print('n_complexity', n_complexity)
        tau = 1.0
        trajectory = Trajectory(n_complexity, model='bandlimited', tau=tau)

        for a_idx, n_anchors in enumerate(p['n_anchors']):
            print('n_anchors', n_anchors)
            environment = Environment(n_anchors)

            for p_idx, n_samples in enumerate(p['n_samples']):
                #  print('n_samples', n_samples)
                times = np.linspace(0, TMAX, n_samples)

                trajectory.set_coeffs(seed=None)
                traj = trajectory.get_trajectory(times=times)

                environment.set_random_anchors(seed=None)
                environment.set_D(traj)

                for noise_idx, noise_sigma in enumerate(p['sigmas_noise']):
                    for m_idx, value_missing in enumerate(p['values_missing']):
                        index_slice = np.s_[c_idx, a_idx, p_idx, noise_idx, m_idx]

                        # set all values to 0 since we have visited them.
                        results['errors'][index_slice] = 0.0
                        results['successes'][index_slice] = 0.0
                        results['num_not_solved'][index_slice] = 0.0

                        for n_it in range(p['n_its']):

                            D_noisy = environment.get_noisy(noise_sigma, seed=None)

                            D_topright = D_noisy[:n_samples, n_samples:].copy()
                            mask = environment.get_mask(p['type_missing'], n=value_missing)
                            D_topright[~mask] = 0.0

                            try:
                                if (solver == None) or (solver == 'semidefRelaxationNoiseless'):
                                    X = semidefRelaxationNoiseless(
                                        D_topright,
                                        environment.anchors, trajectory.basis,
                                        chosen_solver=cvxpy.CVXOPT)
                                    P_hat = X[:DIM, DIM:]
                                elif solver == 'rightInverseOfConstraints':
                                    X = rightInverseOfConstraints(D_topright, environment.anchors,
                                                                  trajectory.basis)
                                    P_hat = X[:DIM, DIM:]
                                elif solver == 'alternativePseudoInverse':
                                    P_hat = alternativePseudoInverse(D_topright, environment.anchors,
                                                                     trajectory.basis)
                                else:
                                    raise ValueError(
                                        'Solver needs to be "semidefRelaxationNoiseless", "rightInverseOfConstraints" or "alternativePseudoInverse"'
                                    )

                                robust_add(results['errors'], index_slice,
                                           np.mean(np.abs(P_hat - trajectory.coeffs)))

                                robust_increment(results['successes'], index_slice)

                            except cvxpy.SolverError:
                                logging.info("could not solve n_samples={}, n_missing={}".format(
                                    n_samples, n_missing))
                                robust_increment(results['num_not_solved'], index_slice)

                            except ZeroDivisionError:
                                logging.info("could not solve.")
                                robust_increment(results['num_not_solved'], index_slice)

                            except np.linalg.LinAlgError:
                                robust_increment(results['num_not_solved'], index_slice)

                        results['errors'][index_slice] = results['errors'][index_slice] / (p['n_its'] - results['num_not_solved'][index_slice])

    if outfolder is not None:
        print('Done with simulation. Saving results...')

        parameters['time'] = int(time.time())

        write_json(outfolder + 'parameters.json', parameters)
        # TODO actually save results as csv file, including results_header.
        save_results(outfolder + 'result_{}_{}', results, results_header)
        return results, results_header
    else:
        return results, results_header


def save_results(filename, results, results_header):
    """ Save results in with increasing number in filename. """
    use_i = 0
    for key, array in results.items():
        for i in range(100):
            try_name = filename.format(key, i) + '.npy'
            if not os.path.exists(try_name):
                np.save(try_name, array, allow_pickle=False)
                print('saved as', try_name)
                break
            else:
                print('exists:', try_name)

    fname = filename.format('header', i) + '.txt'
    with open(fname, 'w') as f:
        for name in results_header:
            f.write(name + '\t')
        print('saved as', fname)


def read_results(filestart):
    """ Read all results saved with above save_results function. """
    results = {}
    dirname = os.path.dirname(filestart)
    results_header = None
    for filename in os.listdir(dirname):
        full_path = os.path.join(dirname, filename)
        if os.path.isfile(full_path) and filestart in full_path:
            print('reading', full_path)
            key = filename.split('_')[-2]
            if key in results.keys():
                new_array = np.load(full_path, allow_pickle=False)
                results[key] += new_array
            elif key == 'header':
                assert full_path[-3:] == 'txt', 'Wrong extension: {}'.format(full_path[-3:])
                results_header = list(np.loadtxt(full_path, dtype='str'))
            else:
                new_array = np.load(full_path, allow_pickle=False)
                print('new key:', key)
                results[key] = new_array
    if results_header is None:
        raise FileNotFoundError('Did not find header file under {} '.format(dirname + '*_header.txt'))
    return results, results_header
