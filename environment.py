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

    def set_random_anchors(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.anchors = 10 * np.random.rand(DIM, self.n_anchors)

    def plot(self):
        plt.scatter(*self.anchors, color='black')
