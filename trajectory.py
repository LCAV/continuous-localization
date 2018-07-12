#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2018 Frederike Duembgen <frederike.duembgen@gmail.com>
#
# Distributed under terms of the MIT license.

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

    def set_random_trajectory(self, seed=None):
        """ Sets self.coeffs, self.Z_opt, self.trajectory. """
        if seed is not None:
            np.random.seed(seed)

        self.coeffs = 5 * \
            np.random.rand(DIM, self.n_complexity)

        self.Z_opt = np.vstack([np.hstack([np.eye(DIM), self.coeffs]),
                                np.hstack([self.coeffs.T, self.coeffs.T @ self.coeffs])])

        self.set_basis()
        self.trajectory = self.coeffs @ self.basis

        return self.trajectory

    def plot(self):
        basis_cont = self.get_basis(n_samples=1000)
        trajectory_cont = self.coeffs @ basis_cont

        plt.scatter(*self.trajectory, color='orange')
        plt.plot(*trajectory_cont, color='orange')
