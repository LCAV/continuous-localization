#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from global_variables import DIM
"""
environment.py: 
"""


class Environment(object):
    def __init__(self, n_anchors=4):
        self.n_anchors = n_anchors
        self.snr = np.inf

    def set_random_anchors(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.anchors = 10 * np.random.rand(DIM, self.n_anchors)

    def plot(self):
        plt.scatter(*self.anchors, color='blue')

    def set_D(self, traj):
        X = np.hstack([traj.trajectory, self.anchors])
        G = X.T @ X
        self.D = np.outer(np.ones(X.shape[1]), np.diag(G)) + np.outer(
            np.diag(G), np.ones(X.shape[1])) - 2 * G
        self.D_true = self.D

    def add_noise(self, noise_sigma, seed, noise_to_square=False):
        self.D = np.copy(self.D_true)

        if seed is not None:
            np.random.seed(seed)

        if noise_sigma>0:
            noise_vector = noise_sigma * np.random.normal(size=self.D.shape)

            if noise_to_square:
                self.D += noise_vector
                snr = np.var(self.D.flatten()) / np.var(noise_vector.flatten())
                self.snr = 10 * np.log10(snr)
            else:
                sqrt_D = np.sqrt(self.D)
                self.D = np.power(sqrt_D + noise_vector, 2)
                snr = np.var(sqrt_D.flatten()) / np.var(noise_vector.flatten())
                self.snr = 10 * np.log10(snr)
        else:
            self.D = self.D_true
            self.snr = np.inf

    def clean_noise(self):
        # resets to the noiseless case
        self.add_noise(0, seed=None)
