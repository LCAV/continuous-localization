# CONTRIBUTION RULES 

### Formatting

Check [Guidelines.md](./GUIDELINES.md) for the detailed formatting rules.

We use automatic `yapf` formatting specified in setup.cfg, and used with yapf version 0.28.0.
You can run:

    ./bin/setup_repository
   
after cloning the repository in order to set up `yapf` formatter
and git hook for removing non important changes from Jupyter Notebooks. 

### Automatic coverage testing

Unused functions, variables, imports, etc. can be quite reliably detected using the *vulture* package. To include the notebooks, make sure to first convert them to python scripts by running 

```
jupyter nbconvert --to=python *.ipynb
```

before checking coverage etc. by running 
```
vulture *.py
```

### Testing notebooks

We use the *frozen* tag for each cell which should not be executed, for example because it entails some heavy computation. This tag is easily added by enabling the nbextension [Freeze](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions/freeze/readme.html).

The following lines of code can be run to install and enable the extension. After installation, 3 buttons appear in the notebook, and the * button 
corresponds to *freezing* a cell.

For pip installation:
```
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
jupyter nbextension enable Freeze
```
or for Conda installation:
```
conda install -c conda-forge jupyter_contrib_nbextensions
jupyter contrib nbextension install --sys-prefix --symlink
jupyter nbextension enable freeze/main
```

You can then test the core notebooks using
```
./bin/run_important_notebooks
```
or test a specific notebook using
```
python bin/run_notebooks.py <notebook name>
```


which will execute all python notebooks while skipping the *frozen* cells. The script will crash if any of the notebooks raise an error.


## Test Suite

To run tests, type in terminal

   pytest test/



### Note on SCS dependency:

The package cvxpy uses SCS as default solver. The problem is that for SCS to work
it needs to be installed with BLAS and LAPACK support, which is not done automatically.
I did not manage to set up SCS as default solver in the travis environment, so I
decided to use CVXOPT instead, for which the installation is less problematic.
