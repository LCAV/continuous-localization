# GUIDELINES

This is an attempt to creating a GUIDELINES file, which
can be used throughout the development of this project to ensure 
format and syntax consistency. 

## Code formatting tools used for this project

For python code, use [yapf](https://github.com/google/yapf) 
with the settings in setup.cfg. 

Run autoformatting of all python files using

```
yapf -i *.py
```

This will read setup.cfg and use the parameters under the tag [yapf].   

Add # yapf: disable in the end of an expression, if you want to keep its formatting.

for whole sections, add `# yapf: disable` and `# yapf: enable` before and after. 

For example: 

```python
Z = {
   (1, 2, 3, 4),
   (5, 6, 7, 8),
   (9, 10, 11, 12),
} # yapf: disable

# yapf: disable

some extremely nicely hand-craft formatted code

# yapf: enable
```

If you would like to autoformat all files using yapf before each local commit, 
simply run the following commands in your local git repository (taken from [here](https://github.com/google/yapf/tree/master/plugins))

```
curl -o pre-commit.sh https://raw.githubusercontent.com/google/yapf/master/plugins/pre-commit.sh
chmod a+x pre-commit.sh
mv pre-commit.sh .git/hooks/pre-commit
```
 
## Documentation format

Use sphynx standard (for docstrings and long comments).

```python

""" This is a short description.

This is a long description.

:param x1: Paramter 1 description.
:param x2: Paramter 1 description.

:returns: List of returnt variables and meaning.

"""
```

## Requirements

Keep track of requirements (including version number where necessary) in the files `requirements.txt` **and** in the
 `environment.yml`. This helps people install requirements both if they use `pip` and `conda`. 


When refering to lab-internal code (or unstable libraries), 
add those as github submodules, for example:

```
git submodule add https://github.com/LCAV/pylocus 
```

## Imports 
Use the correct order of import blocks, and alphabetical order within each block:
1. standard python (e.g. math, sys, time)
2. systemwide standard libraries (e.g. numpy, scipy, pandas)
3. our own libraries (e.g. pylocus)
4. modules from within the project (e.g. trajectory)

## Other 

 - Do not use python shebang and do not put `__main__` in modules &mdash; create a new script if you need. 
 - Save everything in figures/ or results/ (depending on if they are plots or data, respectively)
 - Use `NotImplementError` either for functions that are under development (as in "not yet implemented") or for
  abstract classes. For an option/flag that is not supported use `ValueError`.



