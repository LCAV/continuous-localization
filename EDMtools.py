#!/usr/bin/env python
# module EDMtools

# TODO this module is currently not used. When we actually use these functions, we should
# take them from pylocus instead of using them here.

import numpy as np

import sys
sys.path.append('pylocus/pylocus/')
import algorithms
from mds import MDS


def classicalMDS(EDM_noisy, d):
    print("new MDS")
    return MDS(EDM_noisy, d)

    N = EDM_noisy.shape[0]
    oneN = np.ones(N)
    J = np.eye(N) - np.outer(oneN, oneN) / N
    G = -0.5 * J @ EDM_noisy @ J
    eigVals, eigVecs = np.linalg.eig(G)
    tmp = np.diag(np.sqrt(eigVals[:d]))
    return np.real(np.hstack((tmp, np.zeros([d, N - d]))) @ eigVecs.T)


def procrustes(A, X):
    return algorithms.procrustes(A.T, X.T, scale=False)


def procrustes_adam(A, X):
    # TODO: check if this implementation is more elegant than the one in pylocus.
    # columns of A are the anchors
    # columns of X are the points to be aligned to the anchors
    d = A.shape[0]  # dimension
    L = A.shape[1]  # number anchors
    assert L >= d, 'Procrustes needs at least d anchors.'

    a_bar = np.reshape(np.mean(A, axis=1), (2, 1))
    A_bar = A - a_bar
    x_bar = np.reshape(np.mean(X, axis=1), (2, 1))
    X_bar = X - x_bar

    U, s, VT = np.linalg.svd(X_bar @ A_bar.T, full_matrices=True)
    R = VT.T @ U.T
    oneL = np.ones(L)
    return R @ X_bar + a_bar


if __name__ == "__main__":
    print('nothing happens when running this module.')
