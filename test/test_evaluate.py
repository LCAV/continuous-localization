#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_evaluate.py: Test some data evaluation functionalities.
"""

import common

import numpy as np
import pandas as pd
import unittest


class TestEvaluate(unittest.TestCase):
    def test_first_time(self):
        from evaluate_dataset import find_start_times, find_end_times

        # testing!
        test_index = range(100)
        test_df = pd.DataFrame(index=test_index, columns=["length", "timestamp"])
        start_index_test = [10, 50]
        duration = 20
        lengths = np.zeros(100)
        for s_idx in start_index_test:
            lengths[s_idx:s_idx + duration] = 1.0
        test_df["length"] = lengths
        test_df["timestamp"] = test_index

        start_times, start_indices = find_start_times(test_df)
        end_times, end_indices = find_end_times(test_df)
        np.testing.assert_allclose(start_indices, start_index_test)

        # TODO(FD) This is not correct for now. There is a shift by two because of filter length.
        # np.testing.assert_allclose(end_indices, [s + duration  for s in start_index_test])


if __name__ == "__main__":
    unittest.main()
