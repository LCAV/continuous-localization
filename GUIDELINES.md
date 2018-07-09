This is an attempt to creating a GUIDELINES file, which
can be used throughout the development of this project to ensure 
format and syntax consistency. 

# Code formatting tools used for this project

For python code, use [yapf](https://github.com/google/yapf) 
with the settings in setup.cfg. 

# File headers and footers

Not sure yet what we want to use here. 

# Documentation format

Use sphynx standard: 

```python

""" This is a short description.

This is a long description.

:param x1: Paramter 1 description.
:param x2: Paramter 1 description.

:returns: List of returnt variables and meaning.

"""
```

# Requirements

Keep track of requirements in the file requirements.txt
(including version number where necessary)
