# General TODO list

## To check first
 - [x] Project tango to 2D **before** calculating distances
 - [x] Reconstruct linear trajectory
 - [x] Try different subsets of anchors and measurements

## To fix (and test)
 - [x] Getting times from distances and the `TAU` (issue #36)

## To think about when doing experiments

 - [ ] Use less anchors and/or use bigger space
 - [ ] Measure start position and direction
 - [ ] Start always at the same point and navigate robot to the beginning of 
 the trajectory
 - [ ] Check the circle radius and the speed
 - [ ] Measure the anchors and robot height precisely

## Things to consider 
 - [ ] Subtracting (estimated) noise standard deviation from measurements
 - [ ] Check if noise is better for smaller distances (how?)
 - [ ] Model the trajectory with unknown but constant height
