#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cvxpy
import json
import os
import time

from trajectory import Trajectory
from environment import Environment
from global_variables import DIM
from solvers import OPTIONS, semidefRelaxationNoiseless, rightInverseOfConstraints
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


def run_simulation(parameters, outfolder=None, solver=None):
    """ Run simulation. 

    :param parameters: Can be either the name of the folder where parameters.json is stored, or a new dict of parameters.

    """
    if type(parameters) == str:
        fname = parameters + 'parameters.json'
        parameters = read_params(fname)
        print('read parameters from file {}.'.format(fname))

    elif type(parameters) == dict:
        parameters = parameters

        # if we are trying new parameters and saving in a directroy that already exists,
        # we need to make sure that the saved parameters are actually the same.
        if outfolder is not None:
            try:
                parameters_old = read_params(outfolder + 'parameters.json')
                parameters['time'] = parameters_old['time']
                assert parameters == parameters_old
            except FileNotFoundError:
                print('no conflicting parameters file found.')
            except AssertionError:
                print('Found parameters file with different content than new parameters!')
                raise
    else:
        raise TypeError('parameters needs to be folder name or dictionary.')

    complexities = parameters['complexities']
    anchors = parameters['anchors']
    positions = parameters['positions']
    n_its = parameters['n_its']

    successes = np.full(
        (len(complexities), len(anchors), len(positions), max(positions) * max(anchors)), np.nan)
    num_not_solved = np.full(successes.shape, np.nan)
    num_not_accurate = np.full(successes.shape, np.nan)

    for c_idx, n_complexity in enumerate(complexities):
        print('n_complexity', n_complexity)

        for a_idx, n_anchors in enumerate(anchors):
            print('n_anchors', n_anchors)

            environment = Environment(n_anchors)

            for n_it in range(n_its):
                print('n_it')

                for p_idx, n_positions in enumerate(positions):
                    print('n_positions', n_positions)

                    trajectory = Trajectory(n_positions, n_complexity)

                    trajectory.set_trajectory(seed=None)
                    environment.set_random_anchors(seed=None)
                    environment.set_D(trajectory)

                    # remove some measurements

                    n_measurements = n_positions * n_anchors

                    pairs = np.array(np.meshgrid(range(n_positions), range(n_anchors)))
                    pairs.resize((2, n_positions * n_anchors))
                    for m_idx, n_missing in enumerate(range(n_measurements)):

                        indexes = np.s_[c_idx, a_idx, p_idx, m_idx]

                        # set all values to 0 since we have visited them.
                        if np.isnan(successes[indexes]):
                            successes[indexes] = 0.0
                        if np.isnan(num_not_solved[indexes]):
                            num_not_solved[indexes] = 0.0
                        if np.isnan(num_not_accurate[indexes]):
                            num_not_accurate[indexes] = 0.0

                        #print('n_misisng', n_missing)
                        D_topright = environment.D[:n_positions, n_positions:].copy()
                        indices = np.random.choice(n_measurements, size=n_missing, replace=False)
                        xs = pairs[0, indices]
                        ys = pairs[1, indices]
                        assert len(xs) == n_missing
                        assert len(ys) == n_missing
                        D_topright[xs, ys] = 0.0

                        # assert correct number of missing measurements
                        idx = np.where(D_topright == 0.0)
                        assert n_missing == len(idx[0])

                        try:
                            if (solver == None) or (solver == semidefRelaxationNoiseless):
                                X = semidefRelaxationNoiseless(
                                    D_topright,
                                    environment.anchors,
                                    trajectory.basis,
                                    chosen_solver=cvxpy.CVXOPT)
                            elif solver == 'rightInverseOfConstraints':
                                X = rightInverseOfConstraints(D_topright, environment.anchors,
                                                              trajectory.basis)
                            else:
                                raise ValueError(
                                    'Solver needs to "semidefRelaxationNoiseless" or "rightInverseOfConstraints"'
                                )

                            assert not np.any(np.abs(X[:DIM, DIM:] - trajectory.coeffs) > 1e-10)

                            # TODO: why does this not work?
                            #assert np.testing.assert_array_almost_equal(X[:DIM, DIM:], trajectory.coeffs)

                            robust_increment(successes, indexes)
                        except cvxpy.SolverError:
                            #print("could not solve n_positions={}, n_missing={}".format(n_positions, n_missing))
                            robust_increment(num_not_solved, indexes)
                        except ZeroDivisionError:
                            #print("could not solve n_positions={}, n_missing={}".format(n_positions, n_missing))
                            robust_increment(num_not_solved, indexes)
                        except AssertionError:
                            #print("result not accurate n_positions={}, n_missing={}".format(n_positions, n_missing))
                            robust_increment(num_not_accurate, indexes)
    results = {
        'successes': successes,
        'num-not-solved': num_not_solved,
        'num-not-accurate': num_not_accurate
    }

    if outfolder is not None:
        print('Done with simulation. Saving results...')

        parameters['time'] = time.time()

        if not os.path.exists(outfolder):
            os.makedirs(outfolder)

        save_params(outfolder + 'parameters.json', **parameters)
        save_results(outfolder + 'result_{}_{}.csv', results)
    else:
        return results


def save_results(filename, results):
    """ Save results in with increasing number in filename. """
    for key, array in results.items():
        for i in range(100):
            try_name = filename.format(key, i)
            if not os.path.exists(try_name):
                try_name = filename.format(key, i)
                np.save(try_name, array, allow_pickle=False)
                print('saved as', try_name)
                break
            else:
                print('exists:', try_name)


def read_results(filestart):
    """ Read all results saved with above save_results function. """
    results = {}
    dirname = os.path.dirname(filestart)
    for filename in os.listdir(dirname):
        full_path = os.path.join(dirname, filename)
        if os.path.isfile(full_path) and filestart in full_path:
            print('reading', full_path)
            key = filename.split('_')[-2]
            new_array = np.load(full_path, allow_pickle=False)
            if key in results.keys():
                results[key] += new_array
            else:
                print('new key:', key)
                results[key] = new_array
    return results


def save_params(filename, **kwargs):
    for key in kwargs.keys():
        try:
            kwargs[key] = kwargs[key].tolist()
        except AttributeError as e:
            pass
    with open(filename, 'w') as fp:
        json.dump(kwargs, fp, indent=4)
        print('saved as', filename)


def read_params(filename):
    with open(filename, 'r') as fp:
        param_dict = json.load(fp)
    return param_dict
