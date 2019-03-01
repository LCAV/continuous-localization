# SamplingTrajectories

[![Build Status](https://travis-ci.com/duembgen/SamplingTrajectories.svg?token=VrsjbT3JmKFwqdG5e1cc&branch=master)](https://travis-ci.com/duembgen/SamplingTrajectories)

## Authors

Created by (in alphabetic order):

* Adam Scholefield
* Frederike Duembgen
* Michalina Pacholska

## About

Get this package and its non-standard code dependencies using:

    git clone --recurse-submodules https://github.com/duembgen/SamplingTrajectories.git

Install all standard python requirements using `pip`:

    pip install -r requirements.txt

If you want to use `conda`, you might need to run:

    conda install -c omnia cvxpy
    conda install -c conda-forge cvxopt

before installing the rest of the requirements by running:

    conda install --file requirements.txt

If you want to update the submodule to point to a different branch (i.e. the latest
commit on that branch) then run these lines of code:

    git config -f .gitmodules submodule.pylocus.branch <new branch name>
    git submodule update --remote


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
