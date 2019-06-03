#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import common

import unittest

from simulation import run_simulation


class TestSimulation(unittest.TestCase):
    def test_simulations(self):
        parameters = {
            'key': 'test',
            'n_its': 2,
            'time': 'undefined',
            'positions': [6, 7],
            'complexities': [4, 5],
            'anchors': [3],
            'noise_sigmas': [0],
            'success_thresholds': [0]
        }
        outfolder = 'results/{}/'.format(parameters['key'])
        try:
            run_simulation(parameters, outfolder, solver="rightInverseOfConstraints")
        except RuntimeError as e:
            self.fail("run_simulation raised exception: " + str(e))


if __name__ == "__main__":
    unittest.main()
