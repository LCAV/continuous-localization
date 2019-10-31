#!/bin/bash
wget https://github.com/gtrll/gpslam/raw/master/matlab/data/Plaza1.mat
wget https://github.com/gtrll/gpslam/raw/master/matlab/data/Plaza2.mat
wget http://www.robesafe.es/repository/UAHWiFiDataset/data/uah1.mat
wget http://www.robesafe.es/repository/UAHWiFiDataset/data/uah2.mat
mkdir -p datasets
mv Plaza1.mat Plaza2.mat uah1.mat uah2.mat datasets/
echo "Downloaded Plaza1.mat, Plaza2.mat, uah1.mat and uah2.mat to datasets/ folder."
