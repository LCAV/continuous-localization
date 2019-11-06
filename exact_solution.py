#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exact_solution.py: Find exact solutoin to the optimization problem.  
"""

import numpy as np
import scipy.optimize as opt


def f_multidim(anchors, basis, distance_measurements, coeffs):
    """ 
    :param anchors: anchors dim x N
    :param basis: basis vectors K x M
    :param distance_measurements: matrix of squared distances M x N
    :param coeffs: coefficient matrix dim x K

    :return: vector of differences between estimate distance and measured distance.
    """
    assert basis.shape[0] == coeffs.shape[1]
    assert anchors.shape[0] == coeffs.shape[0]
    assert anchors.shape[1] == distance_measurements.shape[1]
    assert basis.shape[1] == distance_measurements.shape[0]

    X = coeffs.dot(basis)  # is (dim x M)
    diff = anchors[:, :, np.newaxis] - X[:, np.newaxis, :]
    distance_estimates = np.linalg.norm(diff, axis=0)**2
    diff = distance_measurements.T - distance_estimates
    nnz_diffs = diff[distance_measurements.T > 0].flatten()
    return nnz_diffs


def f_onedim(anchors, basis, distance_measurements, coeffs):
    """
    See f_multidim for parameter description

    :return: sum of absolute differences between estimated distances and measured distances.
    """
    return np.sum(np.abs(f_multidim(anchors, basis, distance_measurements, coeffs)))


def objective_ls(coeffs, anchors, basis, distance_measurements):
    """ Wrapper function of above, suitable for scipy.optimize.least_squares."""
    dim = anchors.shape[0]
    K = basis.shape[0]
    coeffs = coeffs.reshape((dim, K))
    return f_multidim(anchors, basis, distance_measurements, coeffs).flatten()


def objective_root(coeffs, anchors, basis, D):
    """ Wrapper function of above, suitable for root finding.
    
    Note that for root finding, the dimension of the output of our objective has to 
    be the same as the dimension of the unknown vector x. Therefore we only keep the first dim*K elements.
    
    """
    dim = anchors.shape[0]
    K = basis.shape[0]
    coeffs = coeffs.reshape((dim, K))
    all_constraints = f_multidim(anchors, basis, D, coeffs)
    if len(all_constraints) > dim * K:
        print('Warning: dropping some of the measurements in objective_root.')
    return all_constraints[:dim * K]
    #return np.full((dim * K, ), f_onedim(anchors, basis, D, coeffs))


def quadratic_constraint(coeffs_vec, anchor, distance, basis):
    coeffs = coeffs_vec.reshape((len(anchor), len(basis)))  # d x K
    return distance - np.linalg.norm(anchor - coeffs.dot(basis))**2


def quadratic_constraint_jac(coeffs_vec, anchor, distance, basis):
    coeffs = coeffs_vec.reshape((len(anchor), len(basis)))  # d x K
    jac = 2 * np.outer(anchor - coeffs.dot(basis), basis.T)  # d x K
    return jac.flatten()


def objective_quadratic(coeffs_vec):
    return 1


def evaluate_constraints(constraints, x):
    max_error = 0
    for constraint in constraints:
        fun = constraint['fun']
        error = fun(x, *constraint['args'])
        max_error = max(abs(error), max_error)

    print('max error constraints:', max_error)


def compute_exact(D_topright, anchors, basis, guess=None, method='least_squares', verbose=False):
    """ Function to compute exact solution to quadratically constrained problem. 

    See also UniquenessStudies.ipynb for seeing how to use these methods. 

    :param D_topright, anchors, basis: see :class:`solvers.semidefRelaxation` for explanation. 
    :param guess: initial guess for coefficients (necessary for some methdos). 
    :param method: Method to use to find exact solution. Currently implemented are 

    - least_squares: do gradient descent on the least squares cost function. We do in total 100
    random initializations in (-10, 10)^KD, and we keep all solutions where the cost function is zero, 
    meaning that all constraints are satisfied.  

    - minimize: We setup a constrained minimization problem, and then try to minimize a banal cost function. It doesn't work well when first guess is not feasible, according to some online discussions. 

    - grid: Do a simple grid search to find zeros of the least-squares cost function. Runs out of memory even for small model sizes. 

    - roots: Find the roots of the cost function. Problem: this only allows us to add exactly K*D constraints, which is almost never enough. 

    :return: list of solutions (least_squares) or single solution. 


    """
    dim = anchors.shape[0]
    K = basis.shape[0]

    if method == 'least_squares':
        kwargs = {'anchors': anchors, 'basis': basis, 'distance_measurements': D_topright}
        max_it = 100
        coeffs_hat_list = []
        for i in range(max_it):
            x0 = 2 * (np.random.rand(dim * K) - 0.5) * 10  # uniform between -10, 10
            sol = opt.least_squares(objective_ls, x0=x0, verbose=0, kwargs=kwargs)

            if np.all(np.abs(sol.fun) < 1e-10):
                new_coeffs_hat = sol.x.reshape((dim, K))
                already_present = False
                for c in coeffs_hat_list:
                    if np.allclose(c, new_coeffs_hat):
                        already_present = True
                        break

                if not already_present:
                    coeffs_hat_list.append(new_coeffs_hat)

        if len(coeffs_hat_list) > 0:
            return coeffs_hat_list
        else:
            raise ValueError('No exact solution found in {} random initializations'.format(max_it))
    elif method == 'minimize':
        if guess is None:
            guess = np.zeros((dim * K, ))

        # set up nonlinear constraints.
        constraints = []
        for m, a_m in enumerate(anchors.T):
            assert len(a_m) == dim
            for n, f_n in enumerate(basis.T):
                dist = D_topright[n, m]
                if dist == 0:
                    continue

                assert len(f_n) == K
                single_args = [a_m, dist, f_n]
                constraint = {
                    'type': 'eq',
                    'fun': quadratic_constraint,
                    'jac': quadratic_constraint_jac,
                    'args': single_args
                }
                constraints.append(constraint)
        constraints_max = constraints[:len(guess)]
        assert len(constraints_max) == len(guess)
        sol = opt.minimize(objective_quadratic, x0=guess, constraints=constraints_max, options={'disp': verbose})

        if sol.success:
            if verbose:
                evaluate_constraints(constraints, sol.x)

            coeffs_hat = sol.x.reshape((dim, K))
            return coeffs_hat
        else:
            raise ValueError('Did not converge with message \n {}'.format(sol.message))
    elif method == 'grid':
        ranges = [slice(-10, 10, 1.0) for _ in range(len(guess))]
        sol = opt.brute(objective_ls, ranges)
        return sol
    elif method == 'roots':
        if guess is None:
            guess = np.zeros((dim * K, ))
        args = (anchors, basis, D_topright)
        sol = opt.root(objective_root, x0=guess, args=args, tol=1e-12)
        if not sol.success:
            raise ValueError('root did not converge with message {}:'.format(sol.message))

        coeffs_hat = sol.x.reshape((dim, K))
        return coeffs_hat
    else:
        raise NotImplementedError('Not implemented:{}'.format(method))


if __name__ == "__main__":
    from measurements import get_measurements, create_anchors
    from trajectory import Trajectory

    trajectory = Trajectory(n_complexity=3, dim=2, model='polynomial')
    np.random.seed(1)
    anchors = create_anchors(dim=2, n_anchors=4)
    trajectory.set_coeffs(seed=1)

    n_samples = 10
    basis, D = get_measurements(trajectory, anchors, n_samples)

    assert np.isclose(f_onedim(anchors, basis, D, trajectory.coeffs), 0)
    assert np.allclose(f_multidim(anchors, basis, D, trajectory.coeffs), 0)

    coeffs_hat = compute_exact(D, anchors, basis)
    np.testing.assert_allclose(coeffs_hat, trajectory.coeffs)
