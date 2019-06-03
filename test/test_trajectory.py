#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import common

import numpy as np
import unittest

from trajectory import Trajectory


class TestTrajectory(unittest.TestCase):
    def setUp(self):
        self.trajectory = Trajectory(n_complexity=3, model='full_bandlimited')
        self.trajectory.coeffs = np.zeros_like(self.trajectory.coeffs)
        self.trajectory.coeffs[0, 2] = 1
        self.trajectory.coeffs[1, 1] = 1

    def test_curvature(self):
        radii, unit_tangents, radial_acc = self.trajectory.get_curvatures(self.trajectory.get_times(n_samples=7))
        np.testing.assert_allclose(np.ones(7), radii)


if __name__ == "__main__":
    unittest.main()
