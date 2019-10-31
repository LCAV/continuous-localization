# CONTRIBUTION RULES 

### Formatting

Check Guidelines.md for the detailed formatting rules.

We use automatic `yapf` formatting specified in setup.cfg, and used with yapf version 0.28.0.
You can run:

    ./scripts/setup_repository
   
after cloning the repository in order to set up `yapf` formatter
and git hook for removing non important changes from Jupyter Notebooks. 

## Test Suite

To run tests, type in terminal

   pytest test/

To see if important notebooks run without error, you can also run

   python runner.py


### Note on SCS dependency:

The package cvxpy uses SCS as default solver. The problem is that for SCS to work
it needs to be installed with BLAS and LAPACK support, which is not done automatically.
I did not manage to set up SCS as default solver in the travis environment, so I
decided to use CVXOPT instead, for which the installation is less problematic.
