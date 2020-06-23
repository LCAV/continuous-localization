# -*- coding: utf-8 -*-
"""
constraints.py: Generate constraints for trajectory_recovery and semidefinite relaxations.

Note that Z is defined as Z = [I_D coeffs; coeffs^T L]
"""

from global_variables import DIM
import numpy as np


def verify_dimensions(distance_matrix, anchors, basis):
    """
    :param distance_matrix: n_positions x n_anchors
    :param anchors: dim x n_anchors
    :param basis: n_complexity x n_positions

    """
    n_positions, n_anchors = distance_matrix.shape
    n_complexity = basis.shape[0]
    dim = anchors.shape[0]

    assert n_positions > n_complexity, 'Cannot compute {} coeffs with only {} measurements.'.format(
        n_complexity, n_positions)
    assert n_anchors > dim, 'Cannot localize in {}D with only {} anchors.'.format(dim, n_anchors)

    assert basis.shape[1] == n_positions, basis.shape
    assert anchors.shape[1] == n_anchors, anchors.shape


def get_extended_constraints(distance_matrix, anchors, basis, vectorized=False, A=None, b=None):
    """ Get constraints on Z given by the distances.

    .. math::
        t_{mn}^T Z t_{mn} = d_{mn}^2

    where :math:`t_{mn} = (a_m^T -f_n^T)^T`

    :param distance_matrix: squared distsance measurements, shape (n_positions x n_anchors)
    :param anchors: anchor coordinates, shape (dim x n_anchors)
    :param basis: basis vectors, shape (n_complexity x n_positions)
    :param A: if given, we append the constraints to A.
    :param b: if given, we append the constraints to b.

    """
    verify_dimensions(distance_matrix, anchors, basis)

    dim = anchors.shape[0]
    Ns, Ms = np.where(distance_matrix > 0)
    n_complexity = basis.shape[0]

    if not vectorized:
        t_mns = []
        D_mns = []
    elif A is None and b is None:
        A = []
        b = []

    for i, (m, n) in enumerate(zip(Ms, Ns)):
        a_m = np.reshape(anchors[:, m], (-1, 1))
        f_n = basis[:, n].reshape(n_complexity, 1)
        t_mn = np.r_[a_m, -f_n]

        if vectorized:
            t_mn = np.array(t_mn)
            tmp = t_mn @ t_mn.T
            A.append(tmp.flatten())
            b.append(distance_matrix[n, m])
        else:
            t_mns.append(t_mn)
            D_mns.append(distance_matrix[n, m])

    if not vectorized:
        return t_mns, D_mns
    else:
        return A, b


def get_constraints_identity(n_complexity, dim=DIM, vectorized=False, A=None, b=None):
    """ Get identity constraints for top left of Z matrix. 

    for not vectorized:

    .. math::
        e_d Z e_{d'} = \delta_{dd'}

    for vectorized: 

    .. math::
        vect(e_{d'}e_{d}^T)  vect(Z) = \delta_{dd'}

    :param A: if given, we append the constraints to A. 
    :param b: if given, we append the constraints to b. 

    """

    if (A is not None or b is not None) and not vectorized:
        raise NotImplementedError

    if not vectorized:
        e_ds = []
        e_dprimes = []
        deltas = []
    elif A is None and b is None:
        A = []
        b = []

    for d in range(dim):
        e_d = np.zeros((dim + n_complexity, 1))
        e_d[d] = 1.0

        for dprime in range(dim):
            e_dprime = np.zeros((dim + n_complexity, 1))
            e_dprime[dprime] = 1.0

            delta = 1.0 if d == dprime else 0.0

            if vectorized:
                tmp = e_dprime @ e_d.T
                A.append(tmp.flatten())
                b.append(delta)
            else:
                e_ds.append(e_d)
                e_dprimes.append(e_dprime)
                deltas.append(delta)

    if not vectorized:
        return e_ds, e_dprimes, deltas
    else:
        return A, b


def get_constraints_symmetry(n_complexity, dim=DIM, vectorized=True, A=None, b=None):
    if not vectorized:
        NotImplementedError
    if vectorized and A is None:
        A = []
    if vectorized and b is None:
        b = []

    for i in range(dim + n_complexity):
        for j in range(i + 1, dim + n_complexity):
            tmp = np.zeros((dim + n_complexity) * (dim + n_complexity))
            tmp[i * (dim + n_complexity) + j] = 1
            tmp[j * (dim + n_complexity) + i] = -1
            if vectorized:
                A.append(tmp.flatten())
                b.append(0)
    return A, b


def get_left_submatrix(idx_a, idx_f, anchors, frame, extended=False):
    """Generated left submatrix (TA) in the basic or in the extended form.

    The extended form means that anchors are extended by 1 making them n_dimensions+1 dimensional.
    Trows of the matrix are tensor product of extended anchors and frame vectors,
    which means that the whole matrix is (n_dimensions+1) * n_constraints x n_measurements dimensional.
    This is because it seems to be a more natural representation - for localising just one point,
    the full matrix is reduced to the (extended) left submatrix.

    :param idx_a: list of anchor indexes for each measurement
    :param idx_f: list of frame indexes for each measurement
    :param anchors: matrix of all available anchors, of size n_dimensions x n_anchors
    :param frame: matrix of all frame vectors, of size n_points x n_constraints
    :param extended: bool, if true then return the extended form

    :return: left part of the constrain matrix of size (n_dimensions) * n_constraints x n_measurements
        or (n_dimensions+1) * n_constraints x n_measurements, if extended
    """

    f_vectors = [frame[:, i] for i in idx_f]
    anchor_vectors = np.concatenate([anchors[:, idx_a], np.ones((1, len(idx_a)))]) if extended else anchors[:, idx_a]
    vectors = [(a[:, np.newaxis] @ f[np.newaxis, :]).flatten() for (f, a) in zip(f_vectors, anchor_vectors.T)]
    return np.array(vectors)


def get_constraints(distance_matrix, anchors, basis, weighted=False):
    """ Return constraints TA, TB, and vector b as defined in paper.

    :param distance_matrix: matrix of square distances, of shape n_positions x n_anchors.
    :param anchors: matrix of all available anchors, of size n_dimensions x n_anchors
    :param basis: matrix of all basis vectors, of size n_constraints x n_positions
    :param weighted: bool, if true return measurements and constraints divided by the weight depended on the distance,
        in order to normalise errors. Makes sense only when errors are added to distances
    """

    verify_dimensions(distance_matrix, anchors, basis)

    Ns, Ms = np.where(distance_matrix > 0)

    T_A = get_left_submatrix(Ms, Ns, anchors, basis, extended=False)
    T_B = get_right_submatrix(Ns, basis, reduced=False)
    b = [(np.sum(anchors[:, m]**2) - distance_matrix[n, m]) / 2 for (n, m) in zip(Ns, Ms)]
    b = np.array(b)

    if weighted:
        weights = np.array([(1.0 / np.sqrt(distance_matrix[n, m] + 1e-1)) for (n, m) in zip(Ns, Ms)])
        T_A = T_A * weights[:, np.newaxis]
        T_B = T_B * weights[:, np.newaxis]
        b = weights * b

    return T_A, T_B, b


def get_right_submatrix(idx_f, frame, reduced=False):
    """Generated right submatrix (TB) in the full or in the reduced form.

    The reduced form means the tensor products of frame vectors are reduced to a basis,
    which means that the size of the submatrix is (n_constraints - 1) x n_measurements.

    :param idx_f: list of frame indexes for each measurement
    :param frame: matrix of all frame vectors, of size n_constraints x n_positions
    :param reduced: bool, if true, the reduced form is returned

    :return: right part of the constrain matrix of size n_constraints**2 x n_measurements.
        or (n_constraints - 1) x n_measurements if reduced is true
    """

    if reduced:
        n_constraints, n_positions = frame.shape
        Ks = np.arange(n_constraints, 2 * n_constraints - 1).reshape((n_constraints - 1, 1))
        Ns = np.arange(n_positions).reshape((n_positions, 1))
        extended_frame = np.cos(Ks @ Ns.T * np.pi / n_positions)
        vectors = [extended_frame[:, idx] for idx in idx_f]
    else:
        vectors = [(frame[:, idx, np.newaxis] @ frame[:, idx, np.newaxis].T).flatten() for idx in idx_f]
    return np.array(vectors)


def get_reduced_constraints(idx_a, idx_f, anchors, frame):
    """ Get full matrix in the reduced form (that can be invertible)

    :param idx_a: list of anchor indexes for each measurement
    :param idx_f: list of frame indexes for each measurement
    :param anchors: matrix of all available anchors, of size n_dimensions x n_anchors
    :param frame: matrix of all frame vectors, of size n_points x n_constraints

    :return: the constrain matrix ((n_dimensions+2) * n_constraints - 1 ) x n_measurements
    """
    return np.concatenate([
        get_left_submatrix(idx_a, idx_f, anchors, frame, extended=True),
        get_right_submatrix(idx_f, frame, reduced=True)
    ],
                          axis=1)


def get_frame(n_constraints, n_positions):  # TODO(michalina) allow random times
    """Generate frame (basis) the fast way, without crating the trajectory.
    It speeds up simulations compared to Trajectory.get_basis.

    :param n_constraints: mode size
    :param n_positions: number of samples

    :return: a frame - an evaluated uniformly basis of bandlimited functions
        of size (n_constraints x n_positions)
    """
    Ks = np.arange(n_constraints).reshape((n_constraints, 1))
    Ns = np.arange(n_positions).reshape((n_positions, 1))
    return np.cos(Ks @ Ns.T * np.pi / n_positions)
