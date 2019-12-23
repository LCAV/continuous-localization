# SamplingTrajectories

[![Build Status](https://travis-ci.com/duembgen/SamplingTrajectories.svg?token=VrsjbT3JmKFwqdG5e1cc&branch=master)](https://travis-ci.com/duembgen/SamplingTrajectories)

## Authors

Created by (in alphabetic order):

* Adam Scholefield
* Frederike Duembgen
* Michalina Pacholska

## Installation

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

Not that for some plotting and saving functionalities, you need to have *LaTeX* installed.

## Contents

### Scripts
- *generate_results*: code to generate the results for different algorithms (see table in paper).

### Notebooks
- *GenerateAllFigures.ipynb*: generate the Figures used in the paper.

### Other
- *datasets/*: Lawnmower and WiFi datasets for range-only localization. See *datasets/README.md* for descriptions.

## Contribute 

If you want to contribute to this repository, please check CONTRIBUTE.md. 

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
