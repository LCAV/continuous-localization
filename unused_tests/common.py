# ! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
common.py: Common routines to be done before all other tests.
"""

import sys

from os.path import abspath, dirname

sys.path.append(dirname(abspath(__file__)) + '/../')
sys.path.append(dirname(abspath(__file__)) + '/../source/')
sys.path.append(dirname(abspath(__file__)) + '/../unused_code/')
