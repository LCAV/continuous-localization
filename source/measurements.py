# -*- coding: utf-8 -*-
"""
measurements.py: Functions to generate measurements from setup. 
"""

import math
import numpy as np

EPS = 1e-10


def add_noise(D, noise_sigma, noise_to_square=False):
    """ Add noise to distances (not squared), leaving out zero distances. """
    D_noisy = np.copy(D)

    if noise_sigma > 0:
        noise_vector = noise_sigma * np.random.normal(size=D.shape)
        # do not add noise to zero(=missing) elements.
        noise_vector[np.abs(D) < EPS] = 0.0

        if noise_to_square:
            D_noisy += noise_vector
        else:
            # add noise to distances, not squares.
            D_noisy[D_noisy > 0] = np.sqrt(D_noisy[D_noisy > 0])
            D_noisy = np.power(D_noisy + noise_vector, 2)
    return D_noisy


def create_anchors(dim, n_anchors, check=False):
    if not check:
        return np.random.rand(dim, n_anchors)
    full_rank = False
    extension = np.ones((1, n_anchors))
    anchors = np.random.rand(dim, n_anchors)
    while not full_rank:
        # check if the extended anchors are linearly independent
        # we would ideally like to check if any subset of anchors
        # of the size n_dimensions + 1 is full rank
        extended = np.concatenate([anchors, extension])
        if np.linalg.matrix_rank(extended) > dim:
            full_rank = True
        else:
            anchors = np.random.rand(dim, n_anchors)
    return anchors


def create_mask(n_samples, n_anchors, strategy, seed=None, verbose=False, **kwargs):
    """ 
    NOT CURRENTLY USED (but can be useful at some point).
    
    Create a mask of shape n_anchors x n_measurements.
    
    :param strategy: strategy to use. Currently implemented:
    - 'uniform': Choose uniformly the n_missing (passed via **kwargs) missing measurements
    - 'single_time': At each time/position there is at most one measurement
    """
    if seed is not None:
        np.random.seed(seed)

    mask = np.empty((n_samples, n_anchors))
    if strategy == 'uniform':
        mask[:, :] = 1.0
        n_missing = kwargs.get('n_missing', 0)

        n_measurements = n_samples * n_anchors
        pairs = np.array(np.meshgrid(range(n_samples), range(n_anchors)))
        pairs.resize((2, n_samples * n_anchors))
        indices = np.random.choice(n_measurements, size=n_missing, replace=False)
        xs = pairs[0, indices]
        ys = pairs[1, indices]
        assert len(xs) == n_missing
        assert len(ys) == n_missing
        mask[xs, ys] = 0.0

        # assert correct number of missing measurements
        idx = np.where(mask == 0.0)
        assert n_missing == len(idx[0])
    elif strategy == 'single_time':
        n_missing = kwargs.get('n_missing', 0)
        if n_samples < n_missing:
            raise ValueError("too many measurements {}<{} requested".format(n_samples, n_missing))
        mask[:, :] = 0.0
        idx_f = np.random.choice(n_samples, n_samples - n_missing, replace=False)
        idx_a = np.random.choice(n_anchors, n_samples - n_missing, replace=True)
        mask[idx_f, idx_a] = 1.0
    else:
        raise NotImplementedError(strategy)

    return mask


def get_D(anchors, samples):
    """ Create squared distance matrix with 

    :param samples: n_positions x dim trajectory points.
    :param anchors: n_anchors x dim anchor points. 

    :return D: matrix of squared distances (n_positions + n_anchors) x (n_positions + n_anchors)
    """
    X = np.hstack([samples, anchors])
    G = X.T @ X
    D = np.outer(np.ones(X.shape[1]), np.diag(G)) + np.outer(np.diag(G), np.ones(X.shape[1])) - 2 * G
    return D


def get_D_topright(anchors, samples):
    n_positions = samples.shape[1]
    D = get_D(anchors, samples)
    return D[:n_positions, n_positions:]


def get_measurements(traj, anchors, seed=None, n_samples=20, times=None):
    """ Get measurements from setup.

    :param traj: Trajectory instance.
    :param anchors: Anchor coordinates, Nxdim.
    :param n_samples: number of samples
    :param seed: random seed

    :return: basis and distance matrix
    """
    # get measurements
    if seed is not None:
        np.random.seed(seed)
    if times is None:
        basis = traj.get_basis(n_samples=n_samples)
    else:
        basis = traj.get_basis(times=times)
    points = traj.get_sampling_points(basis=basis)
    D_topright = get_D_topright(anchors, points)
    return basis, D_topright
