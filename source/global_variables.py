#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
global_variables.py: Some default variable values of project. 
"""

DIM = 2  #: dimension of setup.
TMAX = 1.0  #: for polynomial trajectories: duration of trajectory in seconds. .
TAU = 2.0  #: for bandlimited trajectories: default period in seconds.
MM = 1e-3  #: millimeters
ROBOT_WIDTH = 245.08 * MM  #: current estimate of the robot width
EPSILON = 1e-10  #: a small number used to avoid division by zero whenever needed
ROBOT_HEIGHT = 0.25  #: robot height in meters.

TANGO_SYSTEM_ID = 7585
RTT_SYSTEM_ID = 7592
