# Experiment protocol 

Date: Tuesday 16.07.2019, all day. Measurements from ca. 2.30pm onwards.

## Setup

Most setup was done on Monday 15.07.2019. 

Tango ADF files: Monday1 (rejected too many things moved recording map) and Tuesday. 

- Anchor with ID :da:bc had to be moved a bit to avoid crash. Initial position was: (0.719, 3.484)
final position: (0.560, 3.484)

- Anchors  with ID :d8:05 and :bf:d1 were moved down for the experiments on Tuesday


- The origin is at origin=[3.58, 3.4, 0.0] and rotated by 90 degrees. The coordinates are given by 
`
coord_world = rotation @ (coord_local - origin)
`

## Linear trajectories

Always start at point (1.034, 5.410) (point is marked by pen in the middle back of the robot). 

Move in a straight line (using gcodes poly1,poly2 or poly3 for appropriate lengths) to different endpoints:
- straight6: (5.046, 5.615), also called calib6
- straight5: (4.869, 4.231), also called calib5
- straight4_fail: (?, ?), robot bumped into plug cover. Not used.
- straight4: (5.870, 1.830), not used for calib
- straight3: (4.254, 1.567), also called calib3
- straight2: (2.924, 1.867), also called calib2
- straight1: (1.198, 1.416), also called calib1

## Walking trajectories

- walking_circle1: Walk in a circle.
- walking_circle2: Walk in a circle.
- walking_circle3: Walk in a circle, two rounds.

