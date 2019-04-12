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
        self.params = {'tau': tau}
        self.set_coeffs()

    def copy(self):
        new = Trajectory(self.n_complexity, self.dim, self.model, self.params['tau'])
        new.set_coeffs(coeffs=self.coeffs)
        return new

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
            trajectory = trajectory[:, np.any(mask[:, :] != 0, axis=1)]

        cont_kwargs = {k: val for k, val in kwargs.items() if k != 'marker'}
        plt.plot(*trajectory_cont, **cont_kwargs)
        # avoid having two labels of same thing.
        pop_labels = ['label', 'linestyle']
        for pop_label in pop_labels:
            if pop_label in kwargs.keys():
                kwargs.pop(pop_label)
        plt.scatter(*trajectory, **kwargs)

    def plot_connections(self, basis, anchors, mask, **kwargs):
        trajectory = self.get_sampling_points(basis=basis)
        ns, ms = np.where(mask)
        for n, m in zip(ns, ms):
            p1 = trajectory[:, n]
            p2 = anchors[:, m]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], **kwargs)

    def plot_number_measurements(self, basis, mask=None, legend=False):
        #  mask is n_samples x n_anchors.
        trajectory = self.get_sampling_points(basis=basis)
        if legend:
            label1 = '1'
            label2 = '2'
            label3 = '>2'
        else:
            label1 = label2 = label3 = None
        for i in range(trajectory.shape[1]):
            point = trajectory[:, i]
            if np.sum(mask[i, :]) == 1:
                plt.scatter(*point, color='orange', label=label1)
                label1 = None
            elif np.sum(mask[i, :]) == 2:
                plt.scatter(*point, color='red', label=label2)
                label2 = None
            elif np.sum(mask[i, :]) > 2:
                plt.scatter(*point, color='green', label=label3)
                label3 = None
        if legend:
            plt.legend(title='# measurements')
