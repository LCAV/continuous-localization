#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from global_variables import DIM

"""
trajectory.py: 
"""


class Trajectory(object):
    def __init__(self, n_positions=8, n_complexity=3):
        self.n_positions = n_positions
        self.n_complexity = n_complexity
        self.coeffs = np.zeros(n_complexity)
        self.trajectory = []

    def get_basis(self, n_samples=None):
        if n_samples is None:
            n_samples = self.n_positions

        k = np.reshape(range(self.n_complexity), [self.n_complexity, 1])
        n = np.reshape(np.linspace(0, self.n_positions,
                                   n_samples), [1, n_samples])
        return np.cos(np.pi * k * n / self.n_positions)

    def set_basis(self, n_samples=None):
        """ Sets self.basis. """
        self.basis = self.get_basis(n_samples)

    def set_trajectory(self, seed=None, coeffs=None):
        """ Sets self.coeffs, self.Z_opt, self.trajectory. 
        
        :param coeffs: Matrix of coefficients, DIM X n_complexity. If None, random coeffs are created. 
        :param seed: random seed.
        """
        if seed is not None:
            np.random.seed(seed)

        if coeffs is None:
            self.coeffs = 5 * \
                np.random.rand(DIM, self.n_complexity)
        else:
            self.coeffs = coeffs

        self.Z_opt = np.vstack([np.hstack([np.eye(DIM), self.coeffs]),
                                np.hstack([self.coeffs.T, self.coeffs.T @ self.coeffs])])

        self.set_basis()
        self.trajectory = self.coeffs @ self.basis

        return self.trajectory

    def plot(self, mask=None, **kwargs):
        basis_cont = self.get_basis(n_samples=1000)
        trajectory_cont = self.coeffs @ basis_cont

        if mask is not None:
            trajectory = self.trajectory[:, np.any(
                mask[:self.n_positions, :] != 0, axis=1)]
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
