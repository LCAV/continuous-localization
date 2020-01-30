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


def setup_constraints(c, eps):
    """ In this attempt, I am adding constraints for successful reconstructions
    and the corresponding constraints form failures ('if I remove any measurement, 
    then the reconstruction will fail.. 
    """

    A = []
    b = []

    A.append([-3.0, -3, 1])
    b.append(3 * c)  # success [3, 3, 3]
    A.append([3.0, 3, -1])
    b.append(-2 * c - eps)  # fail [3, 3, 2]

    A.append([-4.0, -3, 1])
    b.append(2 * c)  # success [3, 3, 2, 1]
    A.append([4.0, 2, -1])
    b.append(-2 * c - eps)  # fail [3, 3, 1, 1]

    A.append([-4.0, -3, 1])
    b.append(c)  # success [3, 2, 2, 1]
    A.append([-4.0, -3, 1])
    b.append(0.0)  # success [2, 2, 2, 1]
    A.append([3.0, 3, -1])
    b.append(-eps)  # fail [2, 2, 2]
    A.append([4.0, 2, -1])
    b.append(-eps)  # fail [2, 2, 1, 1]

    A.append([-5.0, -2, 1])
    b.append(0)  # success [2, 2, 1, 1, 1]
    A.append([5.0, 1, -1])
    b.append(-eps)  # fail [2, 1, 1, 1, 1]
    A.append([6.0, 0, -1])
    b.append(-eps)  # fail [1, 1, 1, 1, 1, 1]

    A.append([-7.0, 0, 1])
    b.append(0)  # success [1, 1, 1, 1, 1, 1, 1]

    A.append([-1.0, 1, 0]), b.append(0)  # a >= b
    A.append([0.0, -1, 0]), b.append(-c)  # b >= c

    f = [1, 1, 1]  # minimize K
    return A, b, f


if __name__ == "__main__":
    import sys

    n_complexity = 3
    dim = 2

    # TODO: the results depends on the value of the slack variable epsilon.
    # it shouldn't....
    eps = 0.1

    # TODO chosen like this so that the resulting values for a, b == 1.0.
    c = 0.45

    A, b_vec, f = setup_constraints(c, eps)

    #equality_matrix = [[1, 1, 0]]  # a + b = 2
    #equality_vector = [2]
    #res = linprog(f, A_ub=A, b_ub=b_vec, A_eq=equality_matrix,
    #              b_eq=equality_vector)

    res = linprog(f, A_ub=A, b_ub=b_vec)
    if not res.success:
        print('ERROR: no solution found!')

    print(res)

    a, b, K = res.x
    print('a={:2.2f}, b={:2.2f}, c={:2.2f}, K={:2.2f}'.format(a, b, c, K))

    A = np.array(A)
    x = np.array(res.x)
    b_vec = np.array(b_vec)
    print('constraints satisfied: should all be negative \n', A.dot(x) - b_vec)
