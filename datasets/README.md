# Real data information

## Download instructions

The WiFi and lawnmower datasets can be automatically downloaded by running the script
`download_datasets.sh`. They will be placed in the datasets/ folder. 

## Dataset descriptions

### Drone datasets

- *all folders*: from https://grvc.us.es/staff/felramfab/roslam_datasets/datasets.html.  These datasets are not currently used. 

### Wifi Indoor localization


- *uah1.mat, uah2.mat*: from http://www.robesafe.es/repository/UAHWiFiDataset/. Indoor localization using a phone and WiFi signals to multiple access points of known locations.


### Range-only dataset from Djungash

- *Plaza1.mat, Plaza2.mat*: from https://github.com/gtrll/gpslam/tree/master/matlab/data. Localization of an autonomous lawnmower using UWB-based ranging to access points of known locations.  

The [original website](http://www.frc.ri.cmu.edu/projects/emergencyresponse/RangeData) of this datasets is broken, so we found the alternative source given above. It only has two of the 5 original datasets though, and little description. 

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
