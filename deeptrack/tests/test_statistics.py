# Use this only when running the test locally.
# import sys
# sys.path.append(".")  # Adds the module to path.

import unittest

import numpy as np

from deeptrack import statistics


class TestFeatures(unittest.TestCase):
    
    def test_sum(self):
        input_values = [np.ones((2,)), np.ones((2,))]
        sum_operation = statistics.Sum(axis=0, distributed=False)
        sum_result = sum_operation(input_values)
        self.assertTrue(np.all(sum_result == np.array([2., 2.])))

        input_values = [np.zeros((2, 3)), np.zeros((2, 3))]
        sum_operation = statistics.Sum(axis=1, distributed=False)
        sum_result = sum_operation(input_values)
        expected_result = np.array([[0., 0., 0.], [0., 0., 0.]])
        self.assertTrue(np.all(sum_result == expected_result))

    def test_mean(self):
        input_values = [np.ones((2,)), np.ones((2,))]
        mean_operation = statistics.Mean(axis=0, distributed=False)
        mean_result = mean_operation(input_values)
        self.assertTrue(np.all(mean_result == np.array([1., 1.])))

        input_values = [np.array([1., 2.]), np.array([3., 4.])]
        mean_operation = statistics.Mean(axis=0, distributed=False)
        mean_result = mean_operation(input_values)
        self.assertTrue(np.all(mean_result == np.array([2., 3.])))

    def test_std(self):
        input_values = [np.array([1., 2.]), np.array([1., 3.])]
        std_operation = statistics.Std(axis=0, distributed=False)
        std_result = std_operation(input_values)
        self.assertTrue(np.all(std_result == np.array([0., 0.5])))

    def test_variance(self):
        input_values = [np.array([1., 2.]), np.array([1., 3.])]
        variance_operation = statistics.Variance(axis=0, distributed=False)
        variance_result = variance_operation(input_values)
        self.assertTrue(np.all(variance_result == np.array([0., 0.25])))

    def test_peak_to_peak(self):
        input_values = [np.array([1., 2.]), np.array([1.5, 3.])]
        peak_to_peak_op = statistics.PeakToPeak(axis=0, distributed=False)
        peak_to_peak_result = peak_to_peak_op(input_values)
        self.assertTrue(np.all(peak_to_peak_result == np.array([0.5, 1.])))

    def test_quantile(self):
        input_values = [np.array([1., 2., 3., 1., 10.])]
        quantile_op = statistics.Quantile(q=0.5, axis=1, distributed=False)
        quantile_result = quantile_op(input_values) # median
        self.assertTrue(np.all(quantile_result == np.array([2.])))

    def test_percentile(self):
        input_values = [np.array([1., 2., 3., 4., 10.])]
        percentile_op = statistics.Percentile(q=75, axis=1, distributed=False)
        percentile_result = percentile_op(input_values)
        self.assertTrue(np.all(percentile_result == np.array([4.])))


if __name__ == "__main__":
    unittest.main()