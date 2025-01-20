# Use this only when running the test locally.
# import sys
# sys.path.append(".")  # Adds the module to path.

import unittest

import numpy as np

from deeptrack import statistics, features


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

    def test_prod(self):
        input_values = [np.array([1., 2.]), np.array([3., 4.])]
        prod_operation = statistics.Prod(axis=0, distributed=False)
        prod_result = prod_operation(input_values)
        self.assertTrue(np.all(prod_result == np.array([3., 8.])))

    def test_median(self):
        input_values = [np.array([10., 3., 1., 4., 2.])]
        median_op = statistics.Median(axis=1, distributed=False)
        median_result = median_op(input_values)
        self.assertTrue(np.all(median_result == np.array([3.])))

    def test_cumsum(self):
        input_values = [np.array([1., 2., 3.]), np.array([1., 1., 1.])]
        cumsum_op = statistics.Cumsum(axis=1, distributed=False)
        cumsum_result = cumsum_op(input_values)
        expected_result = np.array([[1., 3., 6.], [1., 2., 3.]])
        self.assertTrue(np.all(cumsum_result == expected_result))

    def test_nan(self):
        input_values = [np.array([1., 2., np.nan]), np.array([np.nan, 1., 1.])]
        mean_op = statistics.Mean(axis=0, distributed=False)
        mean_result = mean_op(input_values)
        self.assertTrue(np.isnan(mean_result[0]))
        self.assertTrue(mean_result[1] == 1.5)
        self.assertTrue(np.isnan(mean_result[2]))

        prod_op = statistics.Prod(axis=0, distributed=False)
        prod_result = prod_op(input_values)
        self.assertTrue(np.isnan(prod_result[0]))
        self.assertTrue(prod_result[1] == 2)
        self.assertTrue(np.isnan(prod_result[2]))

    def test_inf(self):
        input_values = [np.array([1., 2., np.inf]), np.array([np.inf, 1., 1.])]
        mean_op = statistics.Mean(axis=0, distributed=False)
        mean_result = mean_op(input_values)
        self.assertTrue(np.isinf(mean_result[0]))
        self.assertTrue(mean_result[1] == 1.5)
        self.assertTrue(np.isinf(mean_result[2]))

    def test_edge_cases(self):
        edge_cases = [
            -1,
            0,
            1,
            (np.random.rand(3, 5) - 0.5) * 100,
            np.inf,
            np.nan,
            [np.zeros((3, 4)), np.ones((3, 4))],
            np.random.rand(2, 3, 2, 3),
        ]

        all_statistics = [
        statistics.Sum,
        statistics.Mean,
        statistics.Prod,
        statistics.Median,
        statistics.Std,
        statistics.Variance,
        statistics.PeakToPeak,
        statistics.Quantile,
        statistics.Percentile,
    ]
    
        specific_statistics_for_inf = [
            statistics.Sum,
            statistics.Mean,
            statistics.Prod,
            statistics.Median,
        ]

        for case in edge_cases:
            if case is np.inf:
                selected_statistics = specific_statistics_for_inf
            else:
                selected_statistics = all_statistics

            for stat in selected_statistics:
                self._test_single_case(case, stat)

    def _test_single_case(self, case, feature_class):
        feature = feature_class(axis=0, distributed=False)
        result = feature([case])
        self.assertIsNotNone(result)

    def test_broadcast_list(self):
        inp = features.Value([1, 0])
        pipeline = inp - statistics.Mean(inp)
        self.assertListEqual(pipeline(), [0, 0])
        pipeline = inp - (inp >> statistics.Mean())
        self.assertListEqual(pipeline(), [0, 0])


if __name__ == "__main__":
    unittest.main()