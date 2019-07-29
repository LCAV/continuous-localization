# Experiment protocol 

Date: Tuesday 16.07.2019, all day. Measurements from ca. 2.30pm onwards.

## Setup

Most setup was done on Monday 15.07.2019. 

- Tango ADF files (on the Lenovo phone): Monday1 and Tuesday. We used Tuesday in the end because since Monday1 too much had been moved. Note that calibration of Monday1 was not done super carefully because it was mostly used for synchronization. 

- Anchor with ID :da:bc had to be moved a bit to avoid crash. Initial position was: (0.719, 3.484)
final position: (0.560, 3.484). I forgot when exactly it was, sometime in the middle of experiments. The movement was quite small so it should not affect too much. 

- Anchors with ID :d8:05 and :bf:d1 were moved down for the experiments on Tuesday. 

- The robot was measured with respect to the origin_room=[3.58, 3.4, 0.0] and rotated by 90 degrees. For convenience, the coordinates of anchors were measured in the room frame (given by coord_room), with origin in the corner next to the door, x towards the windows and y towards the library. For processing everything is converted to the robot's coordinates as such:
`
coord_robot = rotation @ (coord_room - origin_room)
`

See `convert_room_to_robot` function in evaluate_dataset.py. 

## In-model trajectories

All trajectories (except the pentagone one) were in-model. The robot moved smoothly at an almost constant speed. In the beginning of each dataset the robot was stationary for a few seconds. This data could be used for calibration. 

## In-model trajectories: linear trajectories

The robot did a few linear trajectories, and I measured the start and end point (all given in the room coordinates)

Always start at point (1.034, 5.410) (point is marked by pen in the middle back of the robot). 

Move in a straight line (using gcodes poly1,poly2 or poly3 for appropriate lengths) to different endpoints:
- straight6: (5.046, 5.615), also called calib6
- straight5: (4.869, 4.231), also called calib5
- straight4_fail: (?, ?), robot bumped into plug cover. Not used.
- straight4: (5.870, 1.830), not used for calib
- straight3: (4.254, 1.567), also called calib3
- straight2: (2.924, 1.867), also called calib2
- straight1: (1.198, 1.416), also called calib1

## Out-of-model trajectories

In the end I did a few walking trajectories which are probably out of model, and the pentagon trajectory.

- walking_circle1: Walk in a circle.
- walking_circle2: Walk in a circle.
- walking_circle3: Walk in a circle, two rounds.
- walking: random walking around. 
- stopping: walking in straight lines and stopping between them
- pentagone: make the robot move in a pentagon shape. #TODO need to recover the exact parameters for ground truth.  

