#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from global_variables import DIM
"""
environment.py: 
"""


class Environment(object):
    def __init__(self, n_anchors=4, dim=DIM):
        self.n_anchors = n_anchors
        self.dim = dim
        self.set_random_anchors()

    def set_random_anchors(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.anchors = 10 * np.random.rand(self.dim, self.n_anchors)

    def plot(self):
        plt.scatter(*self.anchors, color='black')

    def annotate(self):
        eps = 0.1
        for i, a in enumerate(self.anchors.T):
            plt.annotate(xy=a[:self.dim] + eps, s='a{}'.format(i))
