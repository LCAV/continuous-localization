#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
baseline_solvers.py: Some baseline solvers which can be used in performance evaluations. 

"""

import numpy as np


def lowRankApproximation(anchors, r):
    """ Return approximation of matrix of rank r. """
    U, s, VT = np.linalg.svd(anchors, full_matrices=False)
    s[r:] = 0
    return U @ np.diag(s) @ VT


def reconstructD_topright(coeffs, basis, anchors):
    """ Construct D_topright from coeffs (coefficients), basis vectors and anchors."""
    N = basis.shape[1]
    M = anchors.shape[1]
    return np.outer(np.ones(N), np.diag(
        anchors.T @ anchors)) - 2 * basis.T @ coeffs.T @ anchors + np.outer(
            np.diag(basis.T @ coeffs.T @ coeffs @ basis), np.ones(M))


def customMDS(D_topright, basis, anchors):
    """ Custom MDS for matrix of shape M, N. """
    [d, M] = anchors.shape
    N = basis.shape[1]
    JM = np.eye(M) - np.ones([M, M]) / M
    tmp = lowRankApproximation(
        JM @ (np.outer(np.diag(anchors.T @ anchors), np.ones(N)) - D_topright.T), d)
    return 0.5 * np.linalg.inv(anchors @ JM @ anchors.T) @ anchors @ tmp @ basis.T @ np.linalg.inv(
        basis @ basis.T)


def SRLS(anchors, basis, coeffs, D_topright):
    """ Return SRLS cost function. """
    D_topright_hat = reconstructD_topright(coeffs, basis, anchors)
    return np.linalg.norm(D_topright - D_topright_hat, 2)


def getSRLSGrad(anchors, basis, coeffs, D_topright):
    """ Get gradient of SRLS function. """
    [K, N] = basis.shape
    M = anchors.shape[1]
    LHS = anchors @ (
        np.outer(np.diag(anchors.T @ anchors), np.ones(N)) - D_topright.transpose()) @ basis.T

    term1 = M * np.outer(np.diag(basis.T @ coeffs.T @ coeffs @ basis), np.ones(K))
    term2 = -2 * basis.T @ coeffs.T @ anchors @ np.outer(np.ones(M), np.ones(K))
    term3 = -D_topright @ np.outer(np.ones(M), np.ones(K))
    RHS = coeffs@basis@((term1+term2+term3)*basis.T)+np.sum(np.diag(anchors.T@anchors))*coeffs@basis@basis.T + \
        anchors@(2*anchors.T@coeffs@basis-np.outer(np.ones(M),
                                                np.diag(basis.T@coeffs.T@coeffs@basis)))@basis.T

    return RHS - LHS


def gradientStep(anchors, basis, coeffs, D_topright, maxIters=10):
    """ Do gradient step for SRLS. """
    grad = getSRLSGrad(anchors, basis, coeffs, D_topright)
    best_cost = SRLS(anchors, basis, coeffs, D_topright)
    coeffs_hat = coeffs
    minStep = 0
    maxStep = 0.01
    for i in range(maxIters):
        step = (maxStep - minStep) / 2
        coeffs_test = coeffs_hat - step * grad
        cost = SRLS(anchors, basis, coeffs_test, D_topright)
        if cost < best_cost:
            best_cost = cost
            coeffs_hat = coeffs_test
            minStep = step
            maxStep = 2 * maxStep
        else:
            maxStep = step
            minStep = minStep / 2
    return coeffs_hat, best_cost


def gradientDescent(anchors, basis, coeffs, D_topright, maxIters=10):
    """ Do finite number of gradient descent steps. """
    coeffs_hat = coeffs
    costs = [SRLS(anchors, basis, coeffs, D_topright)]
    for i in range(maxIters):
        coeffs_hat, cost = gradientStep(anchors, basis, coeffs_hat, D_topright, maxIters=100)
        costs.append(cost)
    return coeffs_hat, costs


def alternateGDandKEonDR(DR_missing, mask, basis, anchors, niter=50, print_out=False, DR_true=None):
    """ Alternate between gradient descent and MDS. """
    d = anchors.shape[0]
    N = basis.shape[1]
    DR_complete = DR_missing.copy()
    DR_complete[mask != 1] = np.mean(DR_complete[mask == 1])

    #DR_complete[DR_complete < 0] = 0.0
    #coeffs = customMDS(DR_complete[:N,:], basis, anchors)
    #coeffs, costs = gradientDescent(anchors,basis,coeffs,DR_complete[:N,:], maxIters=10)
    #DR_complete[:N,:] = reconstructD_topright(coeffs, basis, anchors)

    if DR_true is not None:
        errs = [np.linalg.norm(DR_complete - DR_true)]
    else:
        errs = []

    for i in range(niter):
        # impose matrix rank
        #DR_complete = lowRankApproximation(DR_complete, d+2)

        # impose known entries
        DR_complete[mask == 1] = DR_missing[mask == 1]

        # zero negastive values
        DR_complete[DR_complete < 0] = 0.0

        # approximate coeffiecients
        coeffs = customMDS(DR_complete[:N, :], basis, anchors)
        coeffs, costs = gradientDescent(anchors, basis, coeffs, DR_complete[:N, :], maxIters=10)

        # update DR
        DR_complete[:N, :] = reconstructD_topright(coeffs, basis, anchors)

        if DR_true is not None:
            err = np.linalg.norm(DR_complete - DR_true)
            errs.append(err)
    return DR_complete, errs
