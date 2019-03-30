#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
linprog.py: Linear program to find values of a, b, c  
"""

import numpy as np
from scipy.optimize import linprog


def attempt_one():

    A = [
        [-4, -3, 1],  # success
        [-5, -2, 1],  # success
        [-6, -1, 1],  # success
        [-3, -3, 1],  # success
        [5, 2, -1],  # fail 
        [3, 3, -1],  # fail 
        [-1, 1, 0],  # a > b
        [0, -1, 0],  # b > c
        [0, 0, -1],  # K > K_min
    ]  # require -4*x + -3*y <= -K + 2c etc.
    b = [2 * c, c, 0, 3 * c, 0, 2 * c, 0, -c, -K_min]

    f = [0, 0, 1]  # minimize K
    return A, b, f


def attempt_two():
    eps = 0.5  #0.5

    A = [
        [-3, -3, 1],  # success
        [3, 3, -1],  # fail
        [-4, -3, 1],  # success
        [4, 2, -1],  # fail
        [4, 3, -1],  # fail
        [-5, -2, 1],  # success
        [5, 2, -1],  # fail
        [5, 1, -1],  # fail
        [-1, 1, 0],  # a > b
        [0, -1, 0],  # b > c
        [0, 0, -1]  # K > K_min
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
        0,
        -c,
        -K_min
    ]

    f = [0, 0, 1]  # minimize K
    return A, b, f


if __name__ == "__main__":
    c = 1
    K = 10

    n_complexity = 3
    dim = 2
    K_min = n_complexity * dim
    print(K_min)

    #A, b, f = attempt_one()
    A, b, f = attempt_two()

    res = linprog(f, A_ub=A, b_ub=b)
    print(res)

    a, b, K = res.x
    print('a={}, b={}'.format(a, b))

    A = np.array(A)
    x = np.array(res.x)
    b = np.array(b)
    print(A.dot(x) <= b)
