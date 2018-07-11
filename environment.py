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
environment.py: 
"""

class Environment(object):
    def __init__(self, n_anchors=4):
        self.n_anchors = n_anchors

    def get_random_anchors(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.anchors = 10 * np.random.rand(DIM, self.n_anchors)

    def plot(self):
        plt.scatter(*self.anchors, color='blue')

    def get_measurements(self, traj):
        X = np.hstack([traj.trajectory, self.anchors])
        G = X.T @ X
        D = np.outer(np.ones(X.shape[1]), np.diag(
            G))+np.outer(np.diag(G), np.ones(X.shape[1]))-2*G
        D_tilde = D.copy()
        D_tilde[:traj.n_positions, :traj.n_positions] = 0
        #plt.matshow(D_tilde)
        self.D_topright = D[:traj.n_positions, traj.n_positions:]

    def add_noise(self, noise, seed):
        if seed is not None:
            np.random.seed(seed)
        # TODO: fill in.
