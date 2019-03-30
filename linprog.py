#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
linprog.py: Linear program to empirically find values of a, b, c. 

a is how much a constraint going from the whole plain to a circle is worth. 
b is how much a constraint going from a circle to two points is worth. 
c is how much a constrint going from two points to a single point is worth. 

"""

import numpy as np
from scipy.optimize import linprog

eps = 1e-3  #0.5
c = 1


def attempt_one():
    """ In this attempt, I am only adding constraints for successful reconstructions. """

    A = [
        [-3, -3, 1],  # success
        [-4, -3, 1],  # success
        [-5, -2, 1],  # success
        [-6, -1, 1],  # success
        [5, 2, -1],  # fail 
        [3, 3, -1],  # fail 
        [-1, 1, 0],  # a > b
        [0, -1, 0],  # b > c
    ]  # require -4*x + -3*y <= -K + 2c etc.
    b = [
        3 * c,  # success
        2 * c,  # success
        c,  # success
        0,  # success
        0,  # fail 
        2 * c,  # fail
        0,
        -c
    ]

    f = [0, 0, 1]  # minimize K
    return A, b, f


def attempt_two():
    """ In this attempt, I am adding constraints for successful reconstructions
    and the corresponding constraints form failures ('if I remove any measurement, 
    then the reconstruction will fail.. 
    """

    A = [
        [-3, -3, 1],  # success
        [3, 3, -1],  # fail
        [-4, -3, 1],  # success
        [4, 2, -1],  # fail
        [4, 3, -1],  # fail
        [-5, -2, 1],  # success
        [5, 2, -1],  # fail
        [5, 1, -1],  # fail
        [-6, -1, 1],  # success
        [-1, 1, 0],  # a > b
        [0, -1, 0],  # b > c
    ]
    b = [
        3 * c,  # success
        -2 * c - eps,  # fail
        2 * c,  # success
        -2 * c - eps,  # fail
        -1 * c - eps,  # fail
        c,  # success
        -eps,  # fail
        -c - eps,  # fail
        0,  # success
        -eps,
        -c - eps,
    ]

    f = [0, 0, 1]  # minimize K
    return A, b, f


if __name__ == "__main__":
    import sys

    n_complexity = 3
    dim = 2
    K_min = n_complexity * dim
    print('minimum K:', K_min)

    #A, b, f = attempt_one()
    A, b_vec, f = attempt_two()

    equality_matrix = [[1, 1, 0]]  # a + b = 2
    equality_vector = [2]

    #res = linprog(f, A_ub=A, b_ub=b_vec, A_eq=equality_matrix,
    #              b_eq=equality_vector)
    res = linprog(f, A_ub=A, b_ub=b_vec)
    if not res.success:
        print('no solution found.')
        sys.exit

    a, b, K = res.x
    print('a={:2.2f}, b={:2.2f}, K={:2.2f}'.format(a, b, K))

    A = np.array(A)
    x = np.array(res.x)
    b_vec = np.array(b_vec)
    print('constraints satisfied: should all be negative \n', A.dot(x) - b_vec)
