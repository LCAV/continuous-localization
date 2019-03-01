#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
common.py: Common routines to be done before all other tests. 
"""

def test_prepare():
    import sys
    from os.path import abspath, dirname
    sys.path.append(dirname(abspath(__file__)) + '/../')
