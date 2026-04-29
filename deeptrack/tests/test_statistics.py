import unittest
import warnings

from deeptrack.backend._config import xp
import numpy as np

from deeptrack import statistics, features
from deeptrack.backend import TORCH_AVAILABLE

from deeptrack.tests import BackendTestBase

if TORCH_AVAILABLE:
    import torch

class TestStatistics_NumPy(BackendTestBase):
    BACKEND = "numpy"
    
    def test_sum(self):
        input_values = [xp.ones((2,)), xp.ones((2,))]
        sum_operation = statistics.Sum(axis=0, distributed=False)
        sum_result = sum_operation(input_values)
        self.assertTrue(xp.all(sum_result == xp.asarray([2., 2.])))

        input_values = [xp.zeros((2, 3)), xp.zeros((2, 3))]
        sum_operation = statistics.Sum(axis=1, distributed=False)
        sum_result = sum_operation(input_values)
        expected_result = xp.asarray([[0., 0., 0.], [0., 0., 0.]])
        self.assertTrue(xp.all(sum_result == expected_result))

    def test_mean(self):
        input_values = [xp.ones((2,)), xp.ones((2,))]
        mean_operation = statistics.Mean(axis=0, distributed=False)
        mean_result = mean_operation(input_values)
        self.assertTrue(xp.all(mean_result == xp.asarray([1., 1.])))

        input_values = [xp.asarray([1., 2.]), xp.asarray([3., 4.])]
        mean_operation = statistics.Mean(axis=0, distributed=False)
        mean_result = mean_operation(input_values)
        self.assertTrue(xp.all(mean_result == xp.asarray([2., 3.])))

    def test_std(self):
        input_values = [xp.asarray([1., 2.]), xp.asarray([1., 3.])]
        std_operation = statistics.Std(axis=0, distributed=False)
        std_result = std_operation(input_values)
        self.assertTrue(xp.all(std_result == xp.asarray([0., 0.5])))

    def test_variance(self):
        input_values = [xp.asarray([1., 2.]), xp.asarray([1., 3.])]
        variance_operation = statistics.Variance(axis=0, distributed=False)
        variance_result = variance_operation(input_values)
        self.assertTrue(xp.all(variance_result == xp.asarray([0., 0.25])))

    def test_peak_to_peak(self):
        input_values = [xp.asarray([1., 2.]), xp.asarray([1.5, 3.])]
        peak_to_peak_op = statistics.PeakToPeak(axis=0, distributed=False)
        peak_to_peak_result = peak_to_peak_op(input_values)
        self.assertTrue(xp.all(peak_to_peak_result == xp.asarray([0.5, 1.])))

    def test_quantile(self):
        input_values = [xp.asarray([1., 2., 3., 1., 10.])]
        quantile_op = statistics.Quantile(q=0.5, axis=1, distributed=False)
        quantile_result = quantile_op(input_values) # median
        self.assertTrue(xp.all(quantile_result == xp.asarray([2.])))

    def test_percentile(self):
        input_values = [xp.asarray([1., 2., 3., 4., 10.])]
        percentile_op = statistics.Percentile(q=75, axis=1, distributed=False)
        percentile_result = percentile_op(input_values)
        self.assertTrue(xp.all(percentile_result == xp.asarray([4.])))

    def test_prod(self):
        input_values = [xp.asarray([1., 2.]), xp.asarray([3., 4.])]
        prod_operation = statistics.Prod(axis=0, distributed=False)
        prod_result = prod_operation(input_values)
        self.assertTrue(xp.all(prod_result == xp.asarray([3., 8.])))

    def test_median(self):
        input_values = [xp.asarray([10., 3., 1., 4., 2.])]
        median_op = statistics.Median(axis=1, distributed=False)
        median_result = median_op(input_values)
        self.assertTrue(xp.all(median_result == xp.asarray([3.])))

    def test_cumsum(self):
        input_values = [xp.asarray([1., 2., 3.]), xp.asarray([1., 1., 1.])]
        cumsum_op = statistics.Cumsum(axis=1, distributed=False)
        cumsum_result = cumsum_op(input_values)
        expected_result = xp.asarray([[1., 3., 6.], [1., 2., 3.]])
        self.assertTrue(xp.all(cumsum_result == expected_result))

    def test_nan(self):
        input_values = [xp.asarray([1., 2., xp.nan]), xp.asarray([xp.nan, 1., 1.])]
        mean_op = statistics.Mean(axis=0, distributed=False)
        mean_result = mean_op(input_values)
        self.assertTrue(xp.isnan(mean_result[0]))
        self.assertTrue(mean_result[1] == 1.5)
        self.assertTrue(xp.isnan(mean_result[2]))

        prod_op = statistics.Prod(axis=0, distributed=False)
        prod_result = prod_op(input_values)
        self.assertTrue(xp.isnan(prod_result[0]))
        self.assertTrue(prod_result[1] == 2)
        self.assertTrue(xp.isnan(prod_result[2]))

    def test_inf(self):
        input_values = [xp.asarray([1., 2., xp.inf]), xp.asarray([xp.inf, 1., 1.])]
        mean_op = statistics.Mean(axis=0, distributed=False)
        mean_result = mean_op(input_values)
        self.assertTrue(xp.isinf(mean_result[0]))
        self.assertTrue(mean_result[1] == 1.5)
        self.assertTrue(xp.isinf(mean_result[2]))

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
        # result = feature([case])
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in subtract",
                category=RuntimeWarning,
            )
            result = feature([case])

        self.assertIsNotNone(result)

    def test_broadcast_list(self):
        inp = features.Value([1, 0])
        pipeline = inp - statistics.Mean(inp)
        self.assertListEqual(pipeline(), [0, 0])
        pipeline = inp - (inp >> statistics.Mean())
        self.assertListEqual(pipeline(), [0, 0])

@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed.")
class TestStatistics_Torch(TestStatistics_NumPy):
    BACKEND = "torch"

if __name__ == "__main__":
    unittest.main()