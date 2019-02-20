#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2018 Frederike Duembgen <frederike.duembgen@gmail.com>
#
# Distributed under terms of the MIT license.

import numpy as np
import matplotlib.pyplot as plt

from global_variables import DIM, TMAX

"""
trajectory.py: 
"""

class Trajectory(object):
    """ Create a trajectory of n_positions positions, and complexity n_complexity. """
    def __init__(self, n_complexity=3, model='polynomial', tau=None):
        self.n_complexity = n_complexity
        self.coeffs = None
        self.model = model
        if model == 'bandlimited':
            assert tau is not None, 'Need to specify period tau when creating bandlimited trajectory.'
        self.params = {'tau': tau}

    def set_coeffs(self, coeffs=None, seed=None):
        """ 
        :param coeffs: Matrix of coefficients, DIM x n_complexity. If None (default), random coeffs are created. 
        :param seed: random seed.
        """
        if seed is not None:
            np.random.seed(seed)
        if coeffs is None:
            self.coeffs = 5 * \
                np.random.rand(DIM, self.n_complexity)
        else:
            self.coeffs = coeffs

    def get_basis_vectors(self):
        if self.model == 'polynomial':
            times = np.linspace(0, TMAX, self.n_complexity)
        elif self.model == 'bandlimited':
            times = np.linspace(0, self.params['tau'] / 2.0, self.n_complexity)
        else:
            raise NotImplementedError(self.model); 

    def get_basis(self, times):
        """ Evaluate the basis (polynomial or bandlimited) at the specified time. """ 
        k = np.arange(self.n_complexity)
        self.basis = np.zeros((self.n_complexity, len(times)))
        for n, time in enumerate(times):
            if self.model == 'polynomial':
                self.basis[:, n] = np.power(time, k)
            elif self.model == 'bandlimited':
                self.basis[:, n] = np.cos(2 * np.pi * k * time / self.params['tau'])
            else:
                raise NotImplementedError(self.model); 
        return self.basis

    def get_trajectory(self, times=None):
        """ Evaluate trajectory at given times. """
        if times is None:
            sampling_freq=0.001
            times = np.arange(0, TMAX, sampling_freq)

        trajectory = np.zeros((DIM, len(times)))
        basis = self.get_basis(times=times)
        for n, time in enumerate(times):
            basis_n = basis[:, n]
            position_n = self.coeffs @ basis_n
            trajectory[:, n] = position_n.reshape((-1,))
        return trajectory

    def set_Z_opt(self):
        self.Z_opt = np.vstack([np.hstack([np.eye(DIM), self.coeffs]),
                                np.hstack([self.coeffs.T, self.coeffs.T @ self.coeffs])])

    def plot(self, times=None, **kwargs):
        # plot smooth trajectory.
        if times is None:
            times_cont = np.arange(0, TMAX, 0.001)
        else:
            times_cont = np.linspace(np.min(times), np.max(times), 1000)
        trajectory = self.get_trajectory(times_cont)
        plt.plot(*trajectory, **kwargs)

        # plot a few points only.
        if times is None:
            if self.model == 'bandlimited':
                maxtime = self.params['tau'] / 2.0
            else:
                maxtime = TMAX
            times_sparse = np.linspace(0, maxtime, 10)
        else:
            times_sparse = times
        trajectory_sparse = self.get_trajectory(times_sparse)
        plt.scatter(*trajectory_sparse, **kwargs)

    def plot_number_measurements(self, mask=None, **kwargs):
        times = np.linspace(0, TMAX, n_samples)
        basis_cont = self.get_basis(times=times)
        trajectory_cont = self.coeffs @ basis_cont

        plt.plot(*trajectory_cont)
        for i in range(self.n_positions):
            point = self.trajectory[:, i]
            if np.sum(mask[i, :]) == 1:
                plt.scatter(*point, color='orange')
            if np.sum(mask[i, :]) == 2:
                plt.scatter(*point, color='red')
            if np.sum(mask[i, :]) > 2:
                plt.scatter(*point, color='green')

        plt.scatter(-2.8, 6)
        plt.scatter(-2.8, 5, color='orange')
        plt.scatter(-2.8, 4, color='red')
        plt.scatter(-2.8, 3, color='green')
