#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
environment.py: 
"""

import numpy as np
import matplotlib.pyplot as plt

from global_variables import DIM
from plotting_tools import get_n_colors
from plotting_tools import plot_point_with_name


class Environment(object):
    """ Create environment with anchors. """

    def __init__(self, n_anchors=4):
        self.n_anchors = n_anchors  #: number of anchors
        self.snr = np.inf  #: snr of distances.

    def set_random_anchors(self, seed=None):
        """ Create anchors uniformly distributed in square [0, 10] x [0, 10]. """
        if seed is not None:
            np.random.seed(seed)

        self.anchors = 10 * np.random.rand(DIM, self.n_anchors)

    def plot(self):
        colors = get_n_colors(self.n_anchors)

        i = 1
        for a, col in zip(self.anchors.T, colors):
            plot_point_with_name(a, "$a_{}$".format(i), color=col)
            #plt.scatter(a[0], a[1], color=col)
            i += 1

    def set_D(self, traj):
        """ Get all distances between trajectory and anchors. 

        :param traj: trajectory of shape (DIM x n_times)

        :return: square distance matrix of size (n_times + n_anchors). 
        """
        np.testing.assert_equal(traj.shape[0], self.anchors.shape[0])
        X = np.hstack([traj, self.anchors])
        G = X.T @ X
        self.D = np.outer(np.ones(X.shape[1]), np.diag(G)) + np.outer(
            np.diag(G), np.ones(X.shape[1])) - 2 * G
        self.D_true = self.D
        return self.D_true

    def get_noisy(self, noise_sigma, seed=None, noise_to_square=False):
        """ Get noisy distance matrix. 

        :param noise_sigma: noise to add on distances. 
        :param seed: random seed
        :param noise_to_square: set to True if we add noise to squared distances. 
        
        :return: noisy distance matrix of size n_anchors + x n_anchors. 
        
        """
        D_noisy = np.copy(self.D_true)

        if seed is not None:
            np.random.seed(seed)

        if noise_sigma > 0:
            noise_vector = noise_sigma * np.random.normal(size=D_noisy.shape)

            if noise_to_square:
                D_noisy += noise_vector
            else:
                D_noisy = np.sqrt(self.D)
                D_noisy = np.power(D_noisy + noise_vector, 2)
            # TODO (FD) is this the correct snr calculation?
            snr = np.var(self.D_true.flatten()) / np.var(noise_vector.flatten())
            self.snr = 10 * np.log10(snr)
            return D_noisy
        else:
            #self.snr = np.inf
            return self.D_true

    def get_D_topright(self):
        n_positions = self.D.shape[0] - self.n_anchors
        return self.D[:n_positions, n_positions:]

    def get_mask(self, method, **kwargs):
        """ Create mask for distance matrix. 
        
        :param method: any of uniform, distance-cutoff or n-closest. Depending on method, can give any of the following parameters: 
        :param n: - number of uniformly missing measurements (integer). 
                  - distances bigger than cutoff are dropped (float). 
                  - keep the n-closest distances only (integer).

        :return: boolean mask of shape (n_positions x n_anchors)
        """
        n_positions = self.D.shape[0] - self.n_anchors
        if method == 'uniform':
            mask = np.ones((n_positions, self.n_anchors))

            n_measurements = n_positions * self.n_anchors
            n_missing = kwargs.get('n', 0)
            pairs = np.array(np.meshgrid(range(n_positions), range(self.n_anchors)))
            pairs.resize((2, n_measurements))
            indices = np.random.choice(n_measurements, size=n_missing, replace=False)
            xs = pairs[1, indices]
            ys = pairs[0, indices]
            assert len(xs) == n_missing
            assert len(ys) == n_missing
            mask[ys, xs] = 0.0
            return mask.astype(np.bool)

        D_topright = self.D[:n_positions, n_positions:].copy()
        if method == 'distance-cutoff':
            mask = np.ones((n_positions, self.n_anchors))

            threshold = kwargs.get('n', np.inf)
            mask[D_topright > threshold**2] = 0.0

        elif method == 'n-closest':
            mask = np.zeros((n_positions, self.n_anchors))
            n = kwargs.get('n', 1)

            # return indices of n smallest elements per row.
            idx = np.argpartition(D_topright, n)

            for i in range(n_positions):
                n_closest = idx[i, :n]
                mask[i, n_closest] = 1.0
        return mask.astype(np.bool)

    def plot_measurements(self, traj, mask):
        colors = get_n_colors(self.n_anchors)
        for j, a in enumerate(self.anchors.T):
            for i, t in enumerate(traj.T):
                if mask[i, j] == 0:
                    continue

                plt.plot([a[0], t[0]], [a[1], t[1]], color=colors[j])
