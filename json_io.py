#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import json
import os
"""
json_io.py: Read and write json files in good formats.  
"""


def write_json(filename, param_dict):
    param_writable = {}
    for key, val in param_dict.items():
        # save range as readable format. this avoids having super long lists in the param_dict file.
        if type(val) == range:
            val = str(val)
        elif type(val) == np.ndarray:
            raise TypeError(
                'Parameter {}: use range instead of np.ndarrays instead of processing'.format(key))
        param_writable[key] = val
    with open(filename, 'w') as fp:
        json.dump(param_writable, fp, indent=4)
        print('saved as', filename)


def read_json(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError("File does not exist: {}".format(filename))

    with open(filename, 'r') as fp:
        param_dict = json.load(fp)

    for key, val in param_dict.items():
        # save range as python range instead of string.
        if type(val) == str and val[:5] == "range":
            val = eval(val)
            param_dict[key] = val
    return param_dict
