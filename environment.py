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

    def set_random_anchors(self, bounding_box=None, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.anchors = np.random.rand(self.dim, self.n_anchors)  # between 0, 1
        # 2 x 10

        if bounding_box is not None:
            self.anchors *= np.array(bounding_box)[:, np.newaxis]

    def scale_bounding_box(self, box_dims):
        """Scale anchors to a given size.
        
        :param box_dims: the dimensions of the desired bounding box (x, y), 
        the bounding box is assumed to begin at (0, 0)
        """
        min_point = np.min(self.anchors, axis=1)
        self.anchors -= min_point[:, np.newaxis]
        max_point = np.max(self.anchors, axis=1)
        scale = box_dims / max_point
        self.anchors *= scale[:, np.newaxis]

    def plot(self, **kwargs):
        color = kwargs.pop('color', 'black')
        plt.scatter(*self.anchors, color=color, **kwargs)

    def annotate(self):
        eps = 0.1
        for i, a in enumerate(self.anchors.T):
            plt.annotate(xy=a[:self.dim] + eps, s='a{}'.format(i))
