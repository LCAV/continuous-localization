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

Use sphynx standard: 

```python

""" This is a short description.

This is a long description.

:param x1: Paramter 1 description.
:param x2: Paramter 1 description.

:returns: List of returnt variables and meaning.

"""
```

## Requirements

Keep track of requirements in the file requirements.txt
(including version number where necessary)

When refering to lab-internal code (or unstable libraries), 
add those as github submodules, for example:

```
git submodule add https://github.com/LCAV/pylocus 
```


