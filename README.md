# SamplingTrajectories

[![Build Status](https://travis-ci.com/duembgen/SamplingTrajectories.svg?token=VrsjbT3JmKFwqdG5e1cc&branch=master)](https://travis-ci.com/duembgen/SamplingTrajectories)

## Authors

Created by (in alphabetic order):

* Adam Scholefield
* Frederike Duembgen
* Michalina Pacholska

## Installation

### Basics

Get this package using:

    git clone  https://github.com/duembgen/SamplingTrajectories.git

You can install all standard python requirements it (at least) two ways:
 
1. using `pip` in you favourite Python 3 environment:
    ```
    pip install -r requirements.txt
    ```
2. using `conda`, that will create virtual environment for you:
    ```
    conda env create -f environment.yml
    ```
   
### Note on SCS dependency:

The package cvxpy uses SCS as default solver. The problem is that for SCS to work
it needs to be installed with BLAS and LAPACK support, which is not done automatically.
I did not manage to set up SCS as default solver in the travis environment, so I
decided to use CVXOPT instead, for which the installation is less problematic.

### Contribute 

If you want to contribute to this repository, you should run:

    ./scripts/setup_repository
   
after downloading the repository, in order to set up `yapf` formatter
and git hook for removing non important changes form Jupyter Notebooks. 

## Test Suite

To run tests, type in terminal

   pytest test/

To see if important notebooks run without error, you can also run

   python runner.py

## Documentation

To look at documentation locally, run 

   make html

and then open the file `build/html/index.html` in a browser. 

## License

```
Copyright (c) 2018 Frederike Duembgen, Michalina Pacholska, Adam Scholefield

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
