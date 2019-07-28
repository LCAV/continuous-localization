#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trajectory_creator.py: Creator of used trajectories.
"""

from evaluate_dataset import convert_room_to_robot
from trajectory import Trajectory
import numpy as np

end_points_lines_room = [
    [1.198, 1.416, 0.0],  #1
    [2.924, 1.867, 0.0],  #2
    [4.254, 1.567, 0.0],  #3
    [5.870, 1.830, 0.0],  #4
    [4.869, 4.231, 0.0],  #5
    [5.046, 5.615, 0.0]
]  #6
start_point_room = np.array([1.034, 5.410, 0.0]).reshape((3, 1))


def get_trajectory(file_name):
    if file_name == 'circle2_double.csv':
        trajectory = Trajectory(dim=2, model='full_bandlimited')
        trajectory.set_coeffs(coeffs=np.array([[0, 2, 0], [0, 0, 2]]))
        return trajectory

    elif file_name == 'circle3_triple.csv':
        trajectory = Trajectory(dim=2, model='full_bandlimited')
        trajectory.set_coeffs(coeffs=np.array([[0.1, 1, 0], [1.5, 0, 1]]))
        return trajectory

    elif file_name == 'clover.csv':
        trajectory = Trajectory(dim=2, model='full_bandlimited', n_complexity=5)
        trajectory.set_coeffs(seed=1)
        return trajectory

    elif 'straight' in file_name:  #straight1.csv to straight6.csv
        idx = int(file_name[-5])
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

    else:
        raise NotImplementedError(file_name)
