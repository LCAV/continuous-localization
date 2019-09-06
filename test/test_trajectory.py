#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import common

import numpy as np
import unittest

from trajectory import Trajectory


class TestTrajectory(unittest.TestCase):
    def setUp(self):
        self.n_complexity = 3
        self.trajectory = Trajectory(n_complexity=self.n_complexity, model='full_bandlimited')
        self.trajectory.coeffs = np.zeros_like(self.trajectory.coeffs)
        self.trajectory.coeffs[0, 2] = 2
        self.trajectory.coeffs[1, 1] = 2

    def test_curvature(self):
        radii, unit_tangents, radial_acc = self.trajectory.get_curvatures(self.trajectory.get_times(n_samples=7))
        np.testing.assert_allclose(4 * np.ones(7), radii)

    def test_dimensions(self):
        n_samples = 10
        basis = self.trajectory.get_basis(n_samples=n_samples)
        self.assertEqual(basis.shape, (self.n_complexity, n_samples))

    def test_period_full(self):
        period = self.trajectory.period
        basis = self.trajectory.get_basis(times=[0, period])
        np.testing.assert_almost_equal(basis[:, 0], basis[:, 1])

    def test_period_half(self):
        period = 3
        self.trajectory = Trajectory(n_complexity=3, model='bandlimited', period=period)
        basis = self.trajectory.get_basis(times=[0, period])
        np.testing.assert_almost_equal(basis[:, 0], basis[:, 1])

    def test_times_and_distances_inverse(self):
        n_samples = 10
        times, distances, _ = self.trajectory.get_times_from_distances(n_samples=n_samples)
        times_reconstructed, distances_reconstructed, _ = self.trajectory.get_times_from_distances(
            arbitrary_distances=distances)
        np.testing.assert_almost_equal(distances, distances_reconstructed)
        np.testing.assert_almost_equal(times, times_reconstructed)

    def test_times_and_distances_circle(self):
        distances = [0, 4 * np.pi, 8 * np.pi, 12 * np.pi]
        period = self.trajectory.period
        times, distances_reconstructed, _ = self.trajectory.get_times_from_distances(
            arbitrary_distances=distances, time_steps=100000)
        np.testing.assert_almost_equal(distances, distances_reconstructed)
        np.testing.assert_almost_equal(times, [0, 0.5 * period, period, 1.5 * period], decimal=5)

    def test_times_and_distances_line(self):
        trajectory = Trajectory(n_complexity=2, model='polynomial')
        trajectory.coeffs = np.zeros_like(trajectory.coeffs)
        trajectory.coeffs[0, 1] = 1
        trajectory.coeffs[1, 1] = 1
        distances = [0, np.sqrt(2), 2 * np.sqrt(2)]
        times, distances_reconstructed, _ = trajectory.get_times_from_distances(
            arbitrary_distances=distances, time_steps=100000)
        np.testing.assert_almost_equal(distances, distances_reconstructed)
        np.testing.assert_almost_equal(times, [0, 1, 2], decimal=5)


if __name__ == "__main__":
    unittest.main()
