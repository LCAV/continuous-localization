#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
measurements.py: Functions to generate measurements from setup. 
"""

import math
import numpy as np


def add_noise(D, noise_sigma, noise_to_square=False):
    D_noisy = np.copy(D)

    if noise_sigma > 0:
        noise_vector = noise_sigma * np.random.normal(size=D.shape)

        if noise_to_square:
            D_noisy += noise_vector
        else:
            D_noisy[D_noisy > 0] = np.sqrt(D_noisy[D_noisy > 0])
            D_noisy = np.power(D_noisy + noise_vector, 2)
    return D_noisy


def create_mask(n_samples, n_anchors, strategy, seed=None, verbose=False, **kwargs):
    ''' Create a mask of shape n_anchors x n_measurements. 
    
    :param strategy: strategy to use. Currently implemented: 

    - 'minimal': We randomly delete measures such that we only keep measurements from
    exactly dim+1 anchors, and we have measurements at at least n_complexity different time instances, 
    where dim is the dimension and n_complexity the complexity of the setup. 
    - 'simple': The first point sees D+1 anchors, and the next K-1 points see only anchor 0 and 1. 

    '''
    if seed is not None:
        np.random.seed(seed)

    mask = np.empty((n_samples, n_anchors))

    if strategy == 'minimal':
        mask[:, :] = 0.0
        n_complexity = kwargs.get('n_complexity')
        dim = kwargs.get('dim')
        n_added = kwargs.get('n_added', 0)

        required_samples = dim * n_complexity  # optimal case for quadratic problem
        required_samples = dim * n_complexity + 2 * n_complexity - 1  # optimal case for linearized problem

        required_anchors = dim + 1

        # indices of anchors
        anchors_seen = np.random.choice(n_anchors, size=required_anchors, replace=False)
        # each anchor gets measured by a random sample.
        samples_ = np.random.choice(n_samples, size=required_anchors, replace=True)

        n_already_seen = len(set(samples_))
        if verbose:
            print('already got measurements from {}'.format(set(samples_)))

        samples_missing = set(range(n_samples)).difference(set(samples_))
        samples_missing = list(samples_missing)

        n_missing = required_samples - n_already_seen

        samples_seen = np.random.choice(samples_missing, size=n_missing, replace=False)
        #print('got measurements from {} too'.format(samples_seen))
        anchors_ = np.random.choice(anchors_seen, size=n_missing, replace=True)

        mask[samples_, anchors_seen] = 1.0
        mask[samples_seen, anchors_] = 1.0

        nnz_indices = np.sum(mask > 0)

        samples_nnz, anchors_nnz = np.where(mask)

        assert len(set(anchors_nnz)) == required_anchors
        assert len(set(samples_nnz)) == required_samples

        # randomly add more measurements.
        ns, ms = np.where(mask == 0)
        indices = range(len(ns))
        chosen_indices = np.random.choice(indices, size=n_added, replace=False)
        mask[ns[chosen_indices], ms[chosen_indices]] = 1.0

    elif strategy == 'simple':
        dim = kwargs.get('dim')
        n_complexity = kwargs.get('n_complexity')

        mask[:, :] = 0.0
        mask[0, :dim + 1] = 1.0
        mask[1:n_complexity, :dim] = 1.0
        mask[n_complexity - 1, dim - 1] = 0.0
        assert np.sum(mask) == dim * n_complexity, np.sum(mask)

    elif strategy == 'single':
        mask[:, :] = 0.0
        dim = kwargs.get('dim')

        # the first d+1 points see d+1 different anchors.
        mask[range(dim + 1), range(dim + 1)] = 1.0

        # all following points see exactly one anchor.
        choice = np.arange(mask.shape[1])
        indices = np.random.choice(choice, size=mask.shape[0] - dim - 1, replace=True)
        for i, idx in enumerate(indices):
            mask[dim + 1 + i, idx] = 1.0
        assert np.sum(mask) == mask.shape[0]

    elif strategy == 'uniform':
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

    return mask


def create_mask_in_range(D, range_limit=5):
    ''' Create mask where only points in range are seen.'''
    mask = np.zeros(D.shape)
    mask[np.sqrt(D) <= range_limit] = 1.0
    return mask


def calculate_snr(D, D_noisy):
    noise_vector = D_noisy - D
    snr = np.var(D.flatten()) / np.var(noise_vector.flatten())
    return 10 * np.log10(snr)


def get_D(anchors, samples):
    ''' Create squared distance matrix with 

    :param samples: n_positions x dim trajectory points.
    :return D: matrix of squared distances (n_positions + n_anchors) x (n_positions + n_anchors) #TODO why +?

    '''
    X = np.hstack([samples, anchors])
    G = X.T @ X
    D = np.outer(np.ones(X.shape[1]), np.diag(G)) + np.outer(np.diag(G), np.ones(X.shape[1])) - 2 * G
    return D


def get_D_topright(anchors, samples):
    n_positions = samples.shape[1]
    D = get_D(anchors, samples)
    return D[:n_positions, n_positions:]


def get_measurements(traj, env, seed=None, n_samples=20, noise=None, noise_to_square=False):
    """ Get measurements from setup.

    :param traj: Trajectory instance.
    :param env: Environment instance.
    :param n_samples: number of samples
    :param seed: random seed.
    :param noise: float, noise standard deviation
    :param noise_to_square: bool, if true the noise is added to squared distances, otherwise it is added to distances

    :return: basis and distance matrix
    """
    # get measurements
    if seed is not None:
        np.random.seed(seed)
    basis = traj.get_basis(n_samples=n_samples)
    points = traj.get_sampling_points(basis=basis)
    D_topright = get_D_topright(env.anchors, points)
    if noise is not None:
        D_topright = add_noise(D_topright, noise, noise_to_square)
    return basis, D_topright
