#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trajectory.py: Contains the Trajectory class. 
"""

import numpy as np
import matplotlib.pyplot as plt

from global_variables import DIM, TMAX, TAU


class Trajectory(object):
    """ Trajectory class.

    :member dim: dimension (2 or 3)
    :member n_complexity: complexity of trajectory. 
    :member coeffs: coefficients of trajectory. (dim x n_complexity)
    :member model: trajectory model, either bandlimited or polynomial. 
    """

    def __init__(self, n_complexity=3, dim=DIM, model='bandlimited', tau=TAU):
        self.dim = dim
        self.n_complexity = n_complexity
        self.coeffs = None
        self.model = model
        if model == 'bandlimited':
            assert tau is not None, 'Need to specify period tau when creating bandlimited trajectory.'
        self.params = {'tau': tau}

    def get_times(self, n_samples):
        """ Get times appropriate for this trajectory model. """
        if self.model == 'polynomial':
            times = np.linspace(0, TMAX, n_samples)
        elif self.model == 'bandlimited':
            times = np.linspace(0, self.params['tau'] / 2.0, n_samples)
        else:
            raise NotImplementedError(self.model)

        return times

    def get_basis(self, n_samples=None, times=None):
        """ Get basis vectors evaluated at specific times. 

        :param n_samples: number of samples. 
        :param times: vector of times of length n_samples

        :return: basis vector matrix (n_complexity x n_samples)
        """
        if n_samples is None and times is None:
            raise NotImplementedError('Need to give times or n_samples.')
        elif times is None:
            times = self.get_times(n_samples)
        elif times is not None and n_samples is not None:
            raise AttributeError('Cannot give n_samples and times.')
        elif times is not None and n_samples is None:
            n_samples = len(times)
        else:
            raise NotImplementedError('case not treated:', n_samples, times)

        k = np.reshape(range(self.n_complexity), [self.n_complexity, 1])
        n = np.reshape(times, [1, n_samples])
        if self.model == 'bandlimited':
            return np.cos(2 * np.pi * k * n / self.params['tau'])
        elif self.model == 'polynomial':
            return np.power(n, k)

    def set_coeffs(self, seed=None, coeffs=None):
        if seed is not None:
            np.random.seed(seed)

        if coeffs is None:
            self.coeffs = 5 * \
                np.random.rand(self.dim, self.n_complexity)
        else:
            self.coeffs = coeffs

        dim = self.coeffs.shape[0]
        self.Z_opt = np.vstack(
            [np.hstack([np.eye(dim), self.coeffs]),
             np.hstack([self.coeffs.T, self.coeffs.T @ self.coeffs])])

    def get_sampling_points(self, basis=None, seed=None):
        """ Get points where we get measurements.
        
        """
        points = self.coeffs @ basis
        return points

    def get_continuous_points(self):
        basis_cont = self.get_basis(n_samples=1000)
        trajectory_cont = self.get_sampling_points(basis=basis_cont)
        return trajectory_cont

    def plot(self, basis, mask=None, **kwargs):
        """ Plot continuous and sampled version.

        :param times: times of sampling points.
        :param mask: optional measurements mask (to plot missing measurements)
        :param kwargs: any additional kwargs passed to plt.scatter()

        """

        trajectory_cont = self.get_continuous_points()
        trajectory = self.get_sampling_points(basis=basis)

        if mask is not None:
            trajectory = trajectory[:, np.any(mask[:self.n_positions, :] != 0, axis=1)]

        plt.plot(*trajectory_cont, color='blue')
        plt.scatter(*trajectory, **kwargs)

    def plot_number_measurements(self, basis, mask=None, **kwargs):
        trajectory_cont = self.get_continuous_points()
        trajectory = self.get_sampling_points(basis=basis)

        plt.plot(*trajectory_cont, color='blue')

        for i in range(len(times)):
            point = trajectory[:, i]
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
