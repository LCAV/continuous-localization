#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import os
"""
experiment.py: 
"""


def save_results(filename, results):
    """ Save results in with increasing number in filename. """
    for key, array in results.items():
        for i in range(100):
            try_name = filename.format(key, i)
            if not os.path.exists(try_name):
                try_name = filename.format(key, i)
                np.savetxt(try_name, array, delimiter=',')
                print('saved as', try_name)
                break
            else:
                print('exists:', try_name)


def read_results(filestart):
    """ Read all results saved with above save_results function. """
    results = {}
    dirname = os.path.dirname(filestart)
    for filename in os.listdir(dirname):
        full_path = os.path.join(dirname, filename)
        if os.path.isfile(full_path) and filestart in full_path:
            print('reading', full_path)
            key = filename.split('_')[-2]
            new_array = np.loadtxt(full_path, delimiter=',')
            if key in results.keys():
                results[key] += new_array
            else:
                print('new key:', key)
                results[key] = new_array
    return results


def save_params(filename, **kwargs):
    for key in kwargs.keys():
        try:
            kwargs[key] = kwargs[key].tolist()
        except AttributeError as e:
            pass
    with open(filename, 'w') as fp:
        json.dump(kwargs, fp, indent=4)
        print('saved as', filename)


def read_params(filename):
    with open(filename, 'r') as fp:
        param_dict = json.load(fp)
    return param_dict
