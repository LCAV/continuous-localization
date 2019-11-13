#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
solvers.py: Algorithms used to estimate coefficients from distance measurements. 

"""

import numpy as np
import cvxpy as cp

from constraints import *

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
"""
 Default solver options for CVXOPT.
 Scroll down to "Setting solver options" for explanations here:
 https://www.cvxpy.org/tutorial/advanced/index.html
"""


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


def semidefRelaxation(D_topright, anchors, basis, chosen_solver=cp.SCS, **kwargs):
    """ Solve semidefinite feasibility problem of sensor localization problem. 

    .. centered::
        :math:`Z` =  argmin :math:`eps` 

    .. math::
        s.t. \quad e_d^T  Z  e_{d'} = \delta_{d d'}

        t_i^T  Z  t_i <= di^2 + eps
        t_i^T  Z  t_i <= di^2 - eps

        Z \succeq 0

    In following, N is number of measurements, M is nmber of anchors, dim is dimension 
    and K is trajectory complexity. 

    :param D_topright: N x M
    :param anchors: dim x M
    :param basis: K x N

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
    eps = cp.Variable((1))

    e_ds, e_dprimes, deltas = get_constraints_identity(K)
    t_mns, D_mns = get_constraints_D(D_topright, anchors, basis)

    constraints = []

    for e_d, e_dprime, delta in zip(e_ds, e_dprimes, deltas):
        constraints.append(e_d.T * Z * e_dprime == delta)

    for t_mn, D_topright_mn in zip(t_mns, D_mns):
        t_mn = np.array(t_mn)
        constraints.append(t_mn.T * Z * t_mn <= D_topright_mn + eps)
        constraints.append(t_mn.T * Z * t_mn >= D_topright_mn - eps)

    constraints.append(eps >= 0)

    obj = cp.Minimize(eps)
    prob = cp.Problem(obj, constraints)

    #options['reltol'] = 1e-5
    #options['feastol'] = 1e-5
    prob.solve(solver=chosen_solver, **options)
    print('final tolerance', prob.value)
    return Z.value


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


def trajectory_recovery(D_topright, anchors, basis, average_with_Q=False, weighted=False):
    """ Solve linearised sensor localization problem. 

    First parameters are same as for :func:`.semidefRelaxation`. 

    :param average_with_Q: option to improve noise robustness by averaging the 
                           estimate of P with the knowledge we have for Q=P^TP
    :param weighted: bool, if true use an equivalent of weighted least squares
                    (assuming gaussian noise added to distances)
    """

    dim, M = anchors.shape
    K = basis.shape[0]

    #get constraints
    T_A, T_B, b = get_C_constraints(D_topright, anchors, basis, weighted=weighted)

    Ns, Ms = np.where(D_topright > 0)
    Ns_that_see_an_anchor = len(np.unique(Ns))

    #reduce dimension T_B to its rank
    rankT_B = min(2 * K - 1, Ns_that_see_an_anchor)
    u, s, vh = np.linalg.svd(T_B, full_matrices=False)
    num_zero_SVs = len(np.where(s < 1e-10)[0])
    if len(s) - num_zero_SVs != rankT_B:  #This if can be cut, it was just useful for debugging
        pass
        # raise ValueError('LOGIC ERROR: T_B not of expected rank!!')

    T_B_fullrank = u[:, :rankT_B] @ np.diag(s[:rankT_B])

    T = np.hstack((T_A, -T_B_fullrank / 2))
    #solve with a left-inverse (requires enough measurements - see Thm)
    if T.shape[0] >= T.shape[1]:
        C_hat = np.linalg.inv(T.T @ T) @ T.T @ b
    #solve with a right-inverse if we do not have enough measurements
    else:
        right_inv = T.T @ np.linalg.inv(T @ T.T)
        C_hat = right_inv @ b

    assert len(C_hat) == K * dim + rankT_B

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


def exactSolution(D_topright, anchors, basis, method='grid', verbose=False):
    """ Compute the exact solution.  Just a wrapper of the function compute_exact from exact_solution, added here for consistency.
    """
    from exact_solution import compute_exact
    return compute_exact(D_topright, anchors, basis, method=method, verbose=verbose)


if __name__ == "__main__":
    print('nothing happens when running this module.')
