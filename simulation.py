#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cvxpy

from trajectory import Trajectory
from environment import Environment
from global_variables import DIM
from solvers import OPTIONS, semidefRelaxationNoiseless
from experiment import read_params
"""
run_simulation.py: 
"""


def robust_increment(arr, i, j):
    """ increment value of array if inside bound, and set to 1 if previously nan. """
    if j < arr.shape[1]:
        if np.isnan(arr[i, j]):
            arr[i, j] = 1
        else:
            arr[i, j] += 1


def run_simulation(parameters, outfolder=None):
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

    for n_complexity in complexities:
        print('n_complexity', n_complexity)

        for n_anchors in anchors:
            print('n_anchors', n_anchors)

            # TODO currently working for only one complexity and one anchor.
            successes = np.full((len(positions), max(positions) * n_anchors), np.nan)
            num_not_solved = np.full(successes.shape, np.nan)
            num_not_accurate = np.full(successes.shape, np.nan)

            environment = Environment(n_anchors)

            for n_it in range(n_its):
                print('n_it')

                for i, n_positions in enumerate(positions):
                    print('n_positions', n_positions)

                    trajectory = Trajectory(n_positions, n_complexity)

                    trajectory.set_trajectory(seed=None)
                    environment.set_random_anchors(seed=None)
                    environment.set_D(trajectory)

                    # remove some measurements

                    n_measurements = n_positions * n_anchors

                    pairs = np.array(np.meshgrid(range(n_positions), range(n_anchors)))
                    pairs.resize((2, n_positions * n_anchors))
                    for j, n_missing in enumerate(range(n_measurements)):

                        # set all values to 0 since we have visited them.
                        if np.isnan(successes[i, j]):
                            successes[i, j] = 0.0
                        if np.isnan(num_not_solved[i, j]):
                            num_not_solved[i, j] = 0.0
                        if np.isnan(num_not_accurate[i, j]):
                            num_not_accurate[i, j] = 0.0

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
                            X = semidefRelaxationNoiseless(
                                D_topright,
                                environment.anchors,
                                trajectory.basis,
                                chosen_solver=cvxpy.CVXOPT)

                            assert not np.any(np.abs(X[:DIM, DIM:] - trajectory.coeffs) > 1e-10)

                            # TODO: why does this not work?
                            #assert np.testing.assert_array_almost_equal(X[:DIM, DIM:], trajectory.coeffs)

                            robust_increment(successes, i, j)
                        except cvxpy.SolverError:
                            #print("could not solve n_positions={}, n_missing={}".format(n_positions, n_missing))
                            robust_increment(num_not_solved, i, j)
                        except ZeroDivisionError:
                            #print("could not solve n_positions={}, n_missing={}".format(n_positions, n_missing))
                            robust_increment(num_not_solved, i, j)
                        except AssertionError:
                            #print("result not accurate n_positions={}, n_missing={}".format(n_positions, n_missing))
                            robust_increment(num_not_accurate, i, j)

    results = {
        'successes': successes,
        'num-not-solved': num_not_solved,
        'num-not-accurate': num_not_accurate
    }

    if outfolder is not None:
        print('Done with simulation. Saving results...')

        from experiment import save_params, save_results
        import time
        import os

        parameters['time'] = time.time()

        if not os.path.exists(outfolder):
            os.makedirs(outfolder)

        save_params(outfolder + 'parameters.json', **parameters)
        save_results(outfolder + 'result_{}_{}.csv', results)
    else:
        return results
