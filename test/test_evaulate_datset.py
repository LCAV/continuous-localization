import unittest
import numpy as np
from evaluate_dataset import *


class TestRotation(unittest.TestCase):
    def testRotationToReference2D(self):
        points = np.random.randn(2, 10)
        alpha = 0.3 * np.pi
        rotation = np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
        reference = rotation @ points
        points_rot, (rotation_est, rotation_center, reference_center) = match_reference(reference, points)
        np.testing.assert_array_almost_equal(rotation, rotation_est)
        np.testing.assert_array_almost_equal(reference, points_rot)

    def testRotationToReference3D(self):
        points = np.random.randn(3, 10)
        alpha = 0.6 * np.pi
        rotation = np.array([[1, 0, 0, ], [0, np.cos(alpha), np.sin(alpha)], [0, -np.sin(alpha), np.cos(alpha)]])
        reference = rotation @ points
        points_rot, (rotation_est, _, _) = match_reference(reference, points)
        np.testing.assert_array_almost_equal(rotation, rotation_est)
        np.testing.assert_array_almost_equal(reference, points_rot)

    def testTranslationToReference3D(self):
        points = np.random.randn(3, 10)
        points -= np.mean(points, axis=1)[:, None]
        reference = points + np.array([0, 1, 1])[:, None]
        points_rot, (_, rotation_center, reference_center) = match_reference(reference, points)
        np.testing.assert_array_almost_equal(np.zeros(3), rotation_center)
        np.testing.assert_array_almost_equal(np.array([0, 1, 1]), reference_center)


if __name__ == '__main__':
    unittest.main()
