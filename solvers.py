#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cvxpy as cp

from constraints import *
"""
solvers.py: Algorithms used to estimate coefficients from distance measurements. 

"""

# Default solver options for CVXOPT.
# Scroll down to "Setting solver options" for explanations here:
# https://www.cvxpy.org/tutorial/advanced/index.html
#
# These options can be overwritten from outside (for example in
# python notebook) before calling semidefRelaxation[Noiseless].
OPTIONS = {
    cp.SCS: {
        "verbose": False,
        "max_iters": 2500,
        "eps": 1e-3,  # convergence tolerance
        "alpha": 1.8,  # relaxation parameer
        "scale": 5.0,  # balance between primal and dual residual
        "normalize": True,  # precondition data matrices
        "use_indirect": True,  # use indirect solver for KKT system
    },
    cp.CVXOPT: {
        "verbose": False,
        "max_iters": 100,
        "abstol": 1e-7,
        "reltol": 1e-6,
        "feastol": 1e-7,
        # number of iterative refinement steps after solving KKT.
        "refinement": 1,
        "kktsolver": "chol"  # set to "robust" for LDL fact. without cholesky-preprocessing
    }
}


def semidefRelaxationNoiseless(D_topright, anchors, basis, chosen_solver=cp.SCS, **kwargs):
    """ Solve semidefinite feasibility problem of sensor localization problem. 

    .. centered::
        find :math:`Z` 

    .. math::
        s.t. \quad e_d^T  Z  e_{d'} = \delta_{d d'}

        t_i^T  Z  t_i = di^2

        Z \succeq 0

    parameters are same as for semidefRelaxation. 
    """

    # overwrite predefined options with kwargs.
    options = OPTIONS[chosen_solver]
    options.update(kwargs)

    if options["verbose"]:
        print("Running with options:", OPTIONS[chosen_solver])

    dim, M = anchors.shape
    K = basis.shape[0]
    N = D_topright.shape[0]

    Z = cp.Variable((dim + K, dim + K), PSD=True)

    e_ds, e_dprimes, deltas = get_constraints_identity(K)
    t_mns, D_mns = get_constraints_D(D_topright, anchors, basis)

    constraints = []

    for e_d, e_dprime, delta in zip(e_ds, e_dprimes, deltas):
        constraints.append(e_d.T * Z * e_dprime == delta)

    for t_mn, D_topright_mn in zip(t_mns, D_mns):
        t_mn = np.array(t_mn)
        constraints.append(t_mn.T * Z * t_mn == D_topright_mn)

    obj = cp.Minimize(cp.sum(Z))
    prob = cp.Problem(obj, constraints)

    prob.solve(solver=chosen_solver, **options)
    return Z.value


def semidefRelaxation(D_topright, anchors, basis, chosen_solver=cp.SCS):
    """ Solve semidefinite relaxation of sensor localization problem (SDP). 

    .. centered::
        :math:`Z, D` = argmin :math:`\sum_i(\epsilon_i)`

    .. math::
        s.t. \quad -(d_i, 1) D_i (-d_i, 1)^T = \epsilon_i

        (a_m, -f_n) Z (a_m, -f_n)^T = v_i

        D_i \succeq 0

        Z \succeq 0 
    
    with 

    .. math::
        D_i = (1, u_i; u_i, v_i) \quad u_i^2 = v_i

    :param D_topright: NxM matrix of distance measurements between M anchors and N positions (zero when no measurement available) 
    :param anchors: DxM matrix of anchor positions. 
    :param basis: KxN matrix of basis vectors f_n. 

    :return: Z, the matrix composed of [I, P; P^T, Pbar], of size (dim + K) x (dim + K)
    """
    # TODO this part and the one of the noiseless function could be combined in one.

    print("Running with options:", OPTIONS[chosen_solver])

    K = basis.shape[0]
    dim, M = anchors.shape
    N = D_topright.shape[0]

    constraints = []

    W = D_topright > 0
    Ns, Ms = np.where(W)
    S = len(Ms)

    # We introduce a big matrix consisting of Z and D_mn on the diagonal,
    # to bring the problem into a standard form with only one variable.
    X_size = dim + K + 2 * S
    X = cp.Variable((X_size, X_size), PSD=True)

    # The error that we are minimizing (has to be of form "Variable" even though
    # we are not really interested in its value in the end.
    Epsilon = cp.Variable(shape=(N, M), nonneg=True)

    # DEBUGGING
    constraints.append(X == X.T)

    # impose form
    constraints.append(X[0, 0] == 1.0)
    constraints.append(X[1, 1] == 1.0)
    constraints.append(X[0, 1] == 0.0)

    test = np.ones((X_size, X_size))

    constraints.append(X[dim + K:, :dim + K] == 0.0)
    test[dim + K:, :dim + K] = 0

    for i, (m, n) in enumerate(zip(Ms, Ns)):
        e_n = np.zeros((N, 1))
        e_n[n] = 1.0
        a_m = np.reshape(anchors[:, m], (-1, 1))
        f_n = basis @ e_n
        t_mn = np.r_[a_m, -f_n]

        # express constraint on Z in terms of X.
        b = np.zeros((X_size, 1))
        b[:dim + K] = t_mn
        b[dim + K + 2 * i + 1] = -1.0
        tmp = b @ b.T
        constraints.append(cp.atoms.affine.vec.vec(tmp).T * cp.atoms.affine.vec.vec(X) == 0)
        #constraints.append(b.transpose() * X * b == 0)

        # Express constraint on D_i in terms of X.
        d = np.zeros((X_size, 1))
        d[dim + K + 2 * i] = -D_topright[n, m]  # multiplies 1
        d[dim + K + 2 * i + 1] = 1.0  # multiplies v_i
        tmp = d @ d.T
        constraints.append(
            cp.atoms.affine.vec.vec(tmp).T * cp.atoms.affine.vec.vec(X) == Epsilon[n, m])
        #constraints.append(d.transpose() * X * d == Epsilon[n, m])

        for ys in range(dim + K + 2 * (i + 1), X_size):
            for xs in range(dim + K + 2 * i, dim + K + 2 * (i + 1)):
                constraints.append(X[ys, xs] == 0.0)
                test[ys, xs] = 0.0
    print('Set up constraint {}/{}. Solving...'.format(i + 1, S))

    obj = cp.Minimize(cp.sum(Epsilon))
    prob = cp.Problem(obj, constraints)

    # TODO for debugging only
    #print("Standard form:")
    #print(prob.get_problem_data(chosen_solver))

    prob.solve(solver=chosen_solver, **OPTIONS[chosen_solver])

    if X.value is not None:
        Z = X.value[:dim + K, :dim + K]
        return Z
    else:
        return None


def rightInverseOfConstraints(D_topright, anchors, basis):
    """ Solve linearised sensor localization problem. 

    parameters are same as for semidefRelaxation. 
    """

    dim, M = anchors.shape
    K = basis.shape[0]

    #get constraints
    ConstraintsMat, ConstraintsVec = get_constraints_matrix(D_topright, anchors, basis)
    ConstraintsMat = np.array(ConstraintsMat)
    ConstraintsVec = np.array(ConstraintsVec)

    #apply right inverse
    u, s, vh = np.linalg.svd(ConstraintsMat, full_matrices=False)
    num_zero_SVs = len(np.where(s < 1e-10)[0])
    ConstraintsMat_inv = vh[:-num_zero_SVs, :].T @ np.diag(1 / s[:-num_zero_SVs]) @ u[:, :len(s) - num_zero_SVs].T
    Z_hat = ConstraintsMat_inv @ ConstraintsVec  #right inverse
    Z_hat = Z_hat.reshape([dim + K, dim + K])
    return Z_hat


def alternativePseudoInverse(D_topright, anchors, basis, average_with_Q=False):
    """ Solve linearised sensor localization problem. 

    First parameters are same as for :function:`semidefRelaxation`. 

    :param average_with_Q: option to improve noise robustness by averaging the 
                           estimate of P with the knowledge we have for Q=P^TP
    """

    dim, M = anchors.shape
    K = basis.shape[0]

    #get constraints
    T_A, T_B, b = get_C_constraints(D_topright, anchors, basis)

    Ns, Ms = np.where(D_topright > 0)
    Ns_that_see_an_anchor = len(np.unique(Ns))

    #reduce dimension T_B to its rank
    rankT_B = min(2 * K - 1, Ns_that_see_an_anchor)
    u, s, vh = np.linalg.svd(T_B, full_matrices=False)
    num_zero_SVs = len(np.where(s < 1e-10)[0])
    if len(s) - num_zero_SVs != rankT_B:  #This if can be cut, it was just useful for debugging
        raise ValueError('LOGIC ERROR: T_B not of expected rank!!')

    T_B_fullrank = u[:, :rankT_B] @ np.diag(s[:rankT_B])

    #solve with a left-inverse (requires enough measurements - see Thm)
    T = np.hstack((T_A, -T_B_fullrank / 2))
    C_hat = np.linalg.inv(T.T @ T) @ T.T @ b

    P_hat = C_hat[:dim * K].reshape([dim, K])

    if average_with_Q:
        alpha_hat = P_alpha_hat[dim * K:].reshape([K, K])
        Q_hat1 = P_hat.T @ P_hat
        Q_hat2 = vh[:rankT_B, :].T @ alpha_hat
        Q_hat = Q_hat2 + vh[rankT_B:, :] @ (Q_hat1 - Q_hat2)
        eigVals, eigVecs = np.linalg.eig(Q_hat)
        P_hat = np.diag(np.sqrt(eigVals)) @ eigVecs.T
        #tmp = np.diag(np.sqrt(eigVals[:d]))
        #np.real(np.hstack((tmp, np.zeros([d, N - d]))) @ eigVecs.T)
        #MIGHT NEED TO ENFORCE SYMMETRY ON Q

        #TODO PROJECT ONTO AFFINE SUBSPACE

    return P_hat


def lowRankApproximation(anchors, r):
    """ Return approximation of matrix of rank r. """
    U, s, VT = np.linalg.svd(anchors, full_matrices=False)
    s[r:] = 0
    return U @ np.diag(s) @ VT


def reconstructD_topright(X_0, basis, anchors):
    """ Construct D_topright from X_0 (coefficients), basis vectors and anchors."""
    N = basis.shape[1]
    M = anchors.shape[1]
    return np.outer(np.ones(N), np.diag(
        anchors.T @ anchors)) - 2 * basis.T @ X_0.T @ anchors + np.outer(
            np.diag(basis.T @ X_0.T @ X_0 @ basis), np.ones(M))


def customMDS(D_topright, basis, anchors):
    """ Custom MDS for matrix of shape M, N. """
    [d, M] = anchors.shape
    N = basis.shape[1]
    JM = np.eye(M) - np.ones([M, M]) / M
    tmp = lowRankApproximation(
        JM @ (np.outer(np.diag(anchors.T @ anchors), np.ones(N)) - D_topright.T), d)
    return 0.5 * np.linalg.inv(anchors @ JM @ anchors.T) @ anchors @ tmp @ basis.T @ np.linalg.inv(
        basis @ basis.T)


def SRLS(anchors, basis, X_0, D_topright):
    """ Return SRLS cost function. """
    D_topright_hat = reconstructD_topright(X_0, basis, anchors)
    return np.linalg.norm(D_topright - D_topright_hat, 2)


def getSRLSGrad(anchors, basis, X_0, D_topright):
    """ Get gradient of SRLS function. """
    [K, N] = basis.shape
    M = anchors.shape[1]
    LHS = anchors @ (
        np.outer(np.diag(anchors.T @ anchors), np.ones(N)) - D_topright.transpose()) @ basis.T

    term1 = M * np.outer(np.diag(basis.T @ X_0.T @ X_0 @ basis), np.ones(K))
    term2 = -2 * basis.T @ X_0.T @ anchors @ np.outer(np.ones(M), np.ones(K))
    term3 = -D_topright @ np.outer(np.ones(M), np.ones(K))
    RHS = X_0@basis@((term1+term2+term3)*basis.T)+np.sum(np.diag(anchors.T@anchors))*X_0@basis@basis.T + \
        anchors@(2*anchors.T@X_0@basis-np.outer(np.ones(M),
                                                np.diag(basis.T@X_0.T@X_0@basis)))@basis.T

    return RHS - LHS


def gradientStep(anchors, basis, X_0, D_topright, maxIters=10):
    """ Do gradient step for SRLS. """
    grad = getSRLSGrad(anchors, basis, X_0, D_topright)
    best_cost = SRLS(anchors, basis, X_0, D_topright)
    X_0_hat = X_0
    minStep = 0
    maxStep = 0.01
    for i in range(maxIters):
        step = (maxStep - minStep) / 2
        X_0_test = X_0_hat - step * grad
        cost = SRLS(anchors, basis, X_0_test, D_topright)
        if cost < best_cost:
            best_cost = cost
            X_0_hat = X_0_test
            minStep = step
            maxStep = 2 * maxStep
        else:
            maxStep = step
            minStep = minStep / 2
    return X_0_hat, best_cost


def gradientDescent(anchors, basis, X_0, D_topright, maxIters=10):
    """ Do finite number of gradient descent steps. """
    X_0_hat = X_0
    costs = [SRLS(anchors, basis, X_0, D_topright)]
    for i in range(maxIters):
        X_0_hat, cost = gradientStep(anchors, basis, X_0_hat, D_topright, maxIters=100)
        costs.append(cost)
    return X_0_hat, costs


def alternateGDandKEonDR(DR_missing, mask, basis, anchors, niter=50, print_out=False, DR_true=None):
    """ Alternate between gradient descent and MDS. """
    d = anchors.shape[0]
    N = basis.shape[1]
    DR_complete = DR_missing.copy()
    DR_complete[mask != 1] = np.mean(DR_complete[mask == 1])

    #DR_complete[DR_complete < 0] = 0.0
    #X_0 = customMDS(DR_complete[:N,:], basis, anchors)
    #X_0, costs = gradientDescent(anchors,basis,X_0,DR_complete[:N,:], maxIters=10)
    #DR_complete[:N,:] = reconstructD_topright(X_0, basis, anchors)

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
        X_0 = customMDS(DR_complete[:N, :], basis, anchors)
        X_0, costs = gradientDescent(anchors, basis, X_0, DR_complete[:N, :], maxIters=10)

        # update DR
        DR_complete[:N, :] = reconstructD_topright(X_0, basis, anchors)

        if DR_true is not None:
            err = np.linalg.norm(DR_complete - DR_true)
            errs.append(err)
    return DR_complete, errs


if __name__ == "__main__":
    print('nothing happens when running this module.')
