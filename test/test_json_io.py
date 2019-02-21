#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_json_io.py: 
"""

import sys
sys.path.append('../')

import pytest
import numpy as np
import os

import json_io


def test_exceptions():
    params = {"test": np.arange(4)}
    with pytest.raises(TypeError):
        json_io.write_json("", params)

    testfile = "does_not_exist"
    with pytest.raises(FileNotFoundError):
        json_io.read_json(testfile)


def test_io():
    testfile = "params.test"
    params = {"range": range(4), "string": "test", "float": 0.1, "int": 1}
    json_io.write_json(testfile, params)
    params2 = json_io.read_json(testfile)
    os.remove(testfile)
    assert params == params2
