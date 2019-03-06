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

    :member n_complexity: complexity of trajectory. 
    :member n_positions: Number of positions where we measure. 
    :member coeffs: coefficients of trajectory. (dim x n_complexity)
    :member trajectory: list of points of trajectory (dim x n_positions)
    :member basis: basis vectors evaluated at different time instances. (n_complexity x n_positions)
    :member model: trajectory model, either bandlimited or polynomial. 
    """

    # TODO maybe remove n_positions, trajectory and basis from this class.
    # It should be possible to generate them using an independent vector
    # of "times", of length n_positions.
    def __init__(self, n_positions=8, n_complexity=3, dim=DIM, model='bandlimited', tau=TAU):
        self.n_positions = n_positions
        self.n_complexity = n_complexity
        self.coeffs = np.empty((dim, n_complexity))
        self.trajectory = np.empty((dim, n_positions))
        self.basis = np.empty((n_complexity, n_positions))
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

        :param times: vector of times. 
        :return: basis vector matrix (n_complexity x n_positions)
        """
        if n_samples is None:
            n_samples = self.n_positions

        if times is None:
            times = self.get_times(n_samples)

        k = np.reshape(range(self.n_complexity), [self.n_complexity, 1])
        n = np.reshape(times, [1, n_samples])
        if self.model == 'bandlimited':
            return np.cos(2 * np.pi * k * n / self.params['tau'])
        elif self.model == 'polynomial':
            return np.power(n, k)

    def set_basis(self, n_samples=None, times=None):
        """ Sets self.basis. """
        self.basis = self.get_basis(n_samples, times)

    def set_coeffs(self, seed=None, coeffs=None):
        if seed is not None:
            np.random.seed(seed)

        if coeffs is None:
            self.coeffs = 5 * \
                np.random.rand(*self.coeffs.shape)
        else:
            np.testing.assert_equal(coeffs.shape, self.coeffs.shape)
            self.coeffs = coeffs

        dim = self.coeffs.shape[0]
        self.Z_opt = np.vstack([
            np.hstack([np.eye(dim), self.coeffs]),
            np.hstack([self.coeffs.T, self.coeffs.T @ self.coeffs])
        ])

    def set_trajectory(self, seed=None, coeffs=None, times=None):
        """ Sets self.coeffs, self.Z_opt, self.trajectory. 
        
        :param coeffs: Matrix of coefficients, dim X n_complexity. If None, random coeffs are created. 
        :param seed: random seed.
        """
        self.set_coeffs(seed, coeffs)
        self.set_basis(times=times)
        self.trajectory = self.coeffs @ self.basis

        return self.trajectory

    def plot(self, mask=None, **kwargs):
        basis_cont = self.get_basis(n_samples=1000)
        trajectory_cont = self.coeffs @ basis_cont

        if mask is not None:
            trajectory = self.trajectory[:, np.any(mask[:self.n_positions, :] != 0, axis=1)]
        else:
            trajectory = self.trajectory

        plt.scatter(*self.trajectory, **kwargs)
        plt.plot(*trajectory_cont, **kwargs)

    def plot_number_measurements(self, mask=None, **kwargs):
        basis_cont = self.get_basis(n_samples=1000)
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
