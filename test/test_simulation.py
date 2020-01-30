#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import common

import unittest
import os

from simulation import run_simulation


class TestSimulation(unittest.TestCase):
    def setUp(self):
        self.parameters = {
            'key': 'test',
            'n_its': 2,
            'time': 'undefined',
            'positions': [6, 7],
            'complexities': [4, 5],
            'anchors': [3],
            'noise_sigmas': [0],
            'success_thresholds': [0]
        }
        self.outfolder = 'results/{}/'.format(self.parameters['key'])
        parameters_file = self.outfolder + "/parameters.json"
        if os.path.exists(parameters_file):
            os.remove(parameters_file)

    def test_simulations(self):
        try:
            run_simulation(self.parameters, self.outfolder, solver="trajectory_recovery")
            run_simulation(self.parameters, self.outfolder, solver="weighted_trajectory_recovery")
            run_simulation(self.parameters, self.outfolder, solver="semidef_relaxation_noiseless")
        except RuntimeError as e:
            self.fail("run_simulation raised exception: " + str(e))


if __name__ == "__main__":
    unittest.main()
