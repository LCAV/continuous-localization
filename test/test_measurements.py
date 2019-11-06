# -*- coding: utf-8 -*-

import common

import unittest
from measurements import *


class TestGetAnchors(unittest.TestCase):
    def test_dimensions(self):
        n_anchors = 5
        dims = 2
        self.assertEqual((dims, n_anchors), create_anchors(dims, n_anchors).shape)
        self.assertEqual((dims, n_anchors), create_anchors(dims, n_anchors, check=True).shape)
