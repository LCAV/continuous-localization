## Dataset description

### Cool drone datasets

- All folders

Source: https://grvc.us.es/staff/felramfab/roslam_datasets/datasets.html

These datasets are not currently used. 

### Wifi Indoor localization

- uah1.mat, uah2.mat: from http://www.robesafe.es/repository/UAHWiFiDataset/


### Range-only dataset from Djungash

-Plaza1.mat, Plaza2.mat

https://github.com/gtrll/gpslam/tree/master/matlab/data

### General info

File description (from WiFi website, seems to be similar for Djungash): 
```
GT: Groundtruth path from Laser Scan matching
Time (sec) 	X_pose (m) 	Y_pose (m) 	Heading (rad)

DR: Odometry Input (delta distance traveled and delta heading change)
Time (sec) 	Delta Dist. Trav. (m) 	Delta Heading (rad)

DR_PROC: Dead Reckoned Path from Odometry
Time (sec) 	X_pose (m) 	Y_pose (m) 	Heading (rad)

TL: Surveyed Beacon Locations
Time (sec) 	X_pose (m) 	Y_pose (m)
*NOTE by Frederike: above is probably ID instead of time.

TD:
Time (sec) 	Sender / Antenna ID 	Receiver Node ID 	Range (m)
```
