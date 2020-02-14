#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_coordinate_fitting.py: Test curve fitting.
"""

import common

import unittest

from coordinate_fitting import *

ATOL = 1e-6
RTOL = 1e-6
KWARGS = dict(atol=ATOL, rtol=RTOL)
N_IT = 1


class TestLM(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_fit_trajectory(self):
        for model in ['bandlimited', 'full_bandlimited', 'polynomial']:
            # precision problems for polynomial trajectory
            K = 3 if model == 'polynomial' else 5
            n_samples = 10
            traj = Trajectory(n_complexity=K, dim=2, model=model)
            for i in range(N_IT):
                traj.set_coeffs(seed=i)
                times = traj.get_times(n_samples=n_samples)
                points = traj.get_sampling_points(times=times)

                coeffs = fit_trajectory(points, times, traj)

                # optimization objective
                basis = traj.get_basis(times=times)
                np.testing.assert_allclose(coeffs.dot(basis), points, **KWARGS)
                # optimization variable
                np.testing.assert_allclose(coeffs, traj.coeffs, **KWARGS)

    def test_fit_trajectory_and_times(self):
        for model in ['bandlimited', 'full_bandlimited', 'polynomial']:
            K = 3 if model == 'polynomial' else 5
            n_samples = 10
            traj = Trajectory(n_complexity=K, dim=2, model=model)

            for i in range(N_IT):
                traj.set_coeffs(seed=i)
                traj_times = traj.get_times(n_samples=n_samples)
                points = traj.get_sampling_points(times=traj_times)

                times0 = traj_times
                coeffs, times = fit_trajectory_and_times(points, traj, max_iter=5, times=times0)

                # optimization objective
                basis = traj.get_basis(times=times)
                np.testing.assert_allclose(coeffs.dot(basis), points, **KWARGS)

                # optimization variables
                np.testing.assert_allclose(times, traj_times, **KWARGS)
                np.testing.assert_allclose(coeffs, traj.coeffs, **KWARGS)


if __name__ == "__main__":
    unittest.main()
