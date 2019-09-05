#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trajectory_creator.py: Creator of used trajectories.
"""

from evaluate_dataset import convert_room_to_robot
from global_variables import ROBOT_HEIGHT
from trajectory import Trajectory
import numpy as np

end_points_lines_room = {
    '1': [1.198, 1.416, 0.0],
    '2': [2.924, 1.867, 0.0],
    '3': [4.254, 1.567, 0.0],
    '4': [5.870, 1.830, 0.0],
    '5': [4.869, 4.231, 0.0],
    '6': [5.046, 5.615, 0.0]
}
start_point_room = np.array([1.034, 5.410, 0.0]).reshape((3, 1))


def get_trajectory(filename, dim=2):
    assert (dim == 2) or (dim == 3)

    if 'circle2_double' in filename:
        trajectory = Trajectory(dim=dim, model='full_bandlimited')

        if dim == 2:
            trajectory.set_coeffs(coeffs=np.array([[0, 2, 0], [0, 0, 2]]))
        else:
            trajectory.set_coeffs(coeffs=np.array([[0, 2, 0], [0, 0, 2], [ROBOT_HEIGHT, 0, 0]]))
        return trajectory

    elif 'circle3_triple' in filename:
        trajectory = Trajectory(dim=dim, model='full_bandlimited')
        if dim == 2:
            trajectory.set_coeffs(coeffs=np.array([[0.1, 1, 0], [1.5, 0, 1]]))
        else:
            trajectory.set_coeffs(coeffs=np.array([[0.1, 1, 0], [1.5, 0, 1], [ROBOT_HEIGHT, 0, 0]]))
        return trajectory

    elif 'clover' in filename:
        trajectory = Trajectory(dim=2, model='full_bandlimited', n_complexity=5)
        trajectory.set_coeffs(seed=1)
        return trajectory

    elif 'straight' in filename:  #straight1.csv to straight6.csv
        idx = filename[8]
        end_point_room = np.array(end_points_lines_room[idx]).reshape((3, 1))
        end_point = convert_room_to_robot(end_point_room)
        start_point = convert_room_to_robot(start_point_room)

        trajectory = Trajectory(dim=2, model='polynomial', n_complexity=2)

        #coeffs is dim x K
        direction = end_point - start_point
        direction /= np.linalg.norm(direction)

        #TODO(FD) should scale by velocity here.
        # For using a high velocity is good enough because later we will simply
        # use a certain lenth of the trajectory. therefore we want the velocity
        # to be high enough.
        coeffs = np.c_[start_point, 5 * direction][:2, :]
        trajectory.set_coeffs(coeffs=coeffs)
        return trajectory

    elif 'Plaza1' in filename:
        return Trajectory(dim=2, n_complexity=3, model='full_bandlimited', period=100, full_period=True)

    elif 'Plaza2' in filename:
        return Trajectory(dim=2, n_complexity=3, model='full_bandlimited', period=100.3 - 45.1, full_period=True)

    elif 'uah1' in filename:
        return Trajectory(dim=2, n_complexity=3, model='polynomial')

    elif 'uah2' in filename:
        return Trajectory(dim=2, n_complexity=3, model='full_bandlimited', period=1470, full_period=True)

    else:
        raise NotImplementedError(filename)
