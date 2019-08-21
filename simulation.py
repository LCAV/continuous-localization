#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulation.py: Generate random trajectories and noisy distance estimates, reconstruct trajectory and save errors.  
"""

import numpy as np
import matplotlib.pyplot as plt
import cvxpy
import json
import os
import time
import logging

from environment import Environment
from global_variables import DIM
from measurements import get_measurements, create_mask, add_noise
from solvers import OPTIONS, semidefRelaxationNoiseless, rightInverseOfConstraints, alternativePseudoInverse
from trajectory import Trajectory


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
    if type(parameters) == str:
        fname = parameters + 'parameters.json'
        parameters = read_params(fname)
        print('read parameters from file {}.'.format(fname))

    elif type(parameters) == dict:
        parameters = parameters

        # if we are trying new parameters and saving in a directory that already exists,
        # we need to make sure that the saved parameters are actually the same.
        if outfolder is not None:
            try:
                parameters_old = read_params(outfolder + 'parameters.json')
                parameters['time'] = parameters_old['time']
                assert parameters == parameters_old, 'found conflicting parameters file: {}'.format(outfolder +
                                                                                                    'parameters.json')
            except FileNotFoundError:
                print('no conflicting parameters file found.')
            except AssertionError as error:
                raise (error)
    else:
        raise TypeError('parameters needs to be folder name or dictionary.')

    if 'noise_to_square' not in parameters:
        parameters['noise_to_square'] = False

    if 'measure_distances' not in parameters:
        parameters['measure_distances'] = False

    complexities = parameters['complexities']
    anchors = parameters['anchors']
    positions = parameters['positions']
    n_its = parameters['n_its']
    noise_sigmas = parameters['noise_sigmas']
    success_thresholds = parameters['success_thresholds']
    assert len(success_thresholds) == len(noise_sigmas)

    successes = np.full(
        (len(complexities), len(anchors), len(positions), len(noise_sigmas), max(positions) * max(anchors)), np.nan)
    errors = np.full(successes.shape, np.nan)
    relative_errors = np.full(successes.shape, np.nan)
    absolute_errors = np.full(successes.shape, np.nan)
    num_not_solved = np.full(successes.shape, np.nan)
    num_not_accurate = np.full(successes.shape, np.nan)
    squared_distances = []

    for c_idx, n_complexity in enumerate(complexities):
        print('n_complexity', n_complexity)

        for a_idx, n_anchors in enumerate(anchors):
            print('n_anchors', n_anchors)

            for p_idx, n_positions in enumerate(positions):
                print('n_positions', n_positions)

                n_measurements = n_positions * n_anchors
                for m_idx, n_missing in enumerate(range(n_measurements)):
                    print('measurements idx', m_idx)

                    for noise_idx, noise_sigma in enumerate(noise_sigmas):
                        indexes = np.s_[c_idx, a_idx, p_idx, noise_idx, m_idx]
                        print("noise", noise_sigma)

                        # set all values to 0 since we have visited them.
                        if np.isnan(successes[indexes]):
                            successes[indexes] = 0.0
                        if np.isnan(num_not_solved[indexes]):
                            num_not_solved[indexes] = 0.0
                        if np.isnan(num_not_accurate[indexes]):
                            num_not_accurate[indexes] = 0.0

                        for n_it in range(n_its):

                            trajectory = Trajectory(n_complexity)
                            environment = Environment(n_anchors)
                            trajectory.set_coeffs(seed=None)
                            environment.set_random_anchors(seed=None)

                            basis, D_topright = get_measurements(trajectory, environment, n_samples=n_positions)
                            distances = np.sqrt(D_topright)
                            D_topright = add_noise(D_topright, noise_sigma, parameters["noise_to_square"])
                            mask = create_mask(n_positions, n_anchors, 'uniform', n_missing=n_missing)
                            if parameters['measure_distances']:
                                squared_distances.extend(D_topright.flatten().tolist())
                            D_topright = np.multiply(D_topright, mask)

                            try:
                                if (solver is None) or (solver == semidefRelaxationNoiseless):
                                    X = semidefRelaxationNoiseless(
                                        D_topright, environment.anchors, basis, chosen_solver=cvxpy.CVXOPT)
                                    P_hat = X[:DIM, DIM:]
                                elif solver == 'rightInverseOfConstraints':
                                    X = rightInverseOfConstraints(D_topright, environment.anchors, basis)
                                    P_hat = X[:DIM, DIM:]
                                elif solver == 'alternativePseudoInverse':
                                    P_hat = alternativePseudoInverse(D_topright, environment.anchors, basis)
                                elif solver == 'weightedPseudoInverse':
                                    P_hat = alternativePseudoInverse(
                                        D_topright, environment.anchors, basis, weighted=True)
                                else:
                                    raise ValueError(
                                        'Solver needs to be "semidefRelaxationNoiseless", "rightInverseOfConstraints" or "alternativePseudoInverse"'
                                    )

                                # calculate reconstruction error with respect to distances
                                trajectory_estimated = Trajectory(coeffs=P_hat)
                                _, D_estimated = get_measurements(
                                    trajectory_estimated, environment, n_samples=n_positions)
                                estimated_distances = np.sqrt(D_estimated)

                                robust_add(errors, indexes, np.mean(np.abs(P_hat - trajectory.coeffs)))
                                robust_add(relative_errors, indexes,
                                           np.mean(np.abs(distances - estimated_distances) / (distances + 1e-10)))
                                robust_add(absolute_errors, indexes, np.mean(np.abs(distances - estimated_distances)))

                                assert not np.mean(np.abs(P_hat - trajectory.coeffs)) > success_thresholds[noise_idx]

                                robust_increment(successes, indexes)

                                # TODO: why does this not work?
                                # assert np.testing.assert_array_almost_equal(X[:DIM, DIM:], trajectory.coeffs)

                            except cvxpy.SolverError:
                                logging.info("could not solve n_positions={}, n_missing={}".format(
                                    n_positions, n_missing))
                                robust_increment(num_not_solved, indexes)

                            except ZeroDivisionError:
                                logging.info("could not solve n_positions={}, n_missing={}".format(
                                    n_positions, n_missing))
                                robust_increment(num_not_solved, indexes)

                            except np.linalg.LinAlgError:
                                robust_increment(num_not_solved, indexes)

                            except AssertionError:
                                logging.info("result not accurate n_positions={}, n_missing={}".format(
                                    n_positions, n_missing))
                                robust_increment(num_not_accurate, indexes)

                            errors[indexes] = errors[indexes] / (n_its - num_not_solved[indexes])
                            relative_errors[indexes] = relative_errors[indexes] / (n_its - num_not_solved[indexes])

    results = {
        'successes': successes,
        'num-not-solved': num_not_solved,
        'num-not-accurate': num_not_accurate,
        'errors': errors,
        'relative-errors': relative_errors,
        'absolute-errors': absolute_errors,
        'distances': squared_distances
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
