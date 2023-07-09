import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from logistic_dml import DML
from scipy.special import expit


class TestSplit(unittest.TestCase):
    def test_split(self):
        np.random.seed(0)
        K = 3
        input_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        actual = DML.split(K, input_array)
        expected_type = np.ndarray
        expected_shape = input_array.shape
        self.assertIsInstance(actual, expected_type)
        self.assertEqual(actual.shape, expected_shape)

        # Exactly one observation not assigned a fold
        self.assertTrue(sum((i == 0 for i in actual)) == 1)

        # Remaining observations are assigned to a fold
        for j in [1, 2, 3]:
            self.assertTrue(sum((j == i for i in actual)) == 3)


class TestML(unittest.TestCase):
    def test_ml(self):
        np.random.seed(0)
        # Test case with binary R
        R = np.array([0, 1, 1, 0, 1])
        C = pd.DataFrame({'X1': [1, 4, 7, 10, 13], 'X2': [2, 5, 8, 11, 14], 'X3': [3, 6, 9, 12, 15]})
        Ctest = pd.DataFrame({'X1': [16, 19], 'X2': [17, 20], 'X3': [18, 21]})
        expected = np.array([0.85, 0.9])
        dml = DML(classifier=LogisticRegression())
        np.testing.assert_allclose(np.round(dml.ml(R, C, Ctest), 2), expected, rtol=1e-6)

        # Test case with continuous R
        R = np.array([0.2, 0.5, 0.8])
        C = pd.DataFrame({'X1': [1, 3, 5], 'X2': [2, 4, 6]})
        Ctest = pd.DataFrame({'X1': [7, 9], 'X2': [8, 10]})
        expected = np.array([1.1, 1.4])
        dml = DML(regressor=LinearRegression())
        np.testing.assert_allclose(np.round(dml.ml(R, C, Ctest), 2), expected, rtol=1e-6)

class TestDml(unittest.TestCase):
    def test_dml_linear_regression(self):
        np.random.seed(0)
        Y = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0]*2)
        A = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 1]*2)
        X = pd.DataFrame({'X1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]*2,
                          'X2': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]*2})
        K = 2
        model1 = LogisticRegression()
        model2 = LinearRegression()
        expected_keys = ['mXp', 'rXp']
        expected_mXp_shape = (20,)
        expected_rXp_shape = (20,)
        actual = DML(classifier=model1, regressor=model2).dml(Y, A, X, k_folds=K)
        self.assertIsInstance(actual, dict)
        self.assertEqual(sorted(actual.keys()), expected_keys)
        self.assertEqual(actual['mXp'].shape, expected_mXp_shape)
        self.assertEqual(actual['rXp'].shape, expected_rXp_shape)

    def test_dml_logistic_regression(self):
        """The following test fails and it would fail for the R code if Liu et al used logistic
        regression in their code. Perhaps logistic regression should not be used to fit Wp values
        """
        np.random.seed(0)
        Y = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0]*2)
        A = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 1]*2)
        X = pd.DataFrame({'X1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]*2,
                          'X2': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]*2})
        K = 2
        model1 = LogisticRegression()
        model2 = LinearRegression()
        expected_keys = ['mXp', 'rXp']
        expected_mXp_shape = (20,)
        expected_rXp_shape = (20,)
        actual = DML(classifier=model1, regressor=model2).dml(Y, A, X, k_folds=K)
        self.assertIsInstance(actual, dict)
        self.assertEqual(sorted(actual.keys()), expected_keys)
        self.assertEqual(actual['mXp'].shape, expected_mXp_shape)
        self.assertEqual(actual['rXp'].shape, expected_rXp_shape)


class TestEstimate(unittest.TestCase):
    def test_estimate(self):
        """Nb: This unit test replicates a unit test in R."""
        Y = np.array([1, 0, 1, 1, 0])
        A = np.array([2.3, 1.2, 3.4, 2.1, 1.8])

        dml = {
            'mXp': [0.2, 0.3, 0.1, 0.4, 0.3],
            'rXp': [0.5, 0.4, 0.6, 0.3, 0.2]
        }
        actual = DML().estimate_beta(Y, A, dml)
        self.assertAlmostEqual(actual, 0.29689899)


class TestBootstrap(unittest.TestCase):
    def test_bootstrap(self):
        """Nb: This unit test replicates a unit test in R."""
        np.random.seed(0)
        Y = np.array([1, 0, 0, 1, 1])
        A = np.array([2, 1, 0, 1, 2])

        dml = {
            'rXp': [0, 0, 0, 0, 0],
            'mXp': [0, 0, 0, 0, 0]
        }
        actual = DML().bootstrap(Y, A, dml, 2000)
        lb, ub = actual[0], actual[1]
        mean = actual[2]
        sd = actual[3]
        self.assertLess(mean, 0.9)
        self.assertGreater(mean, 0.7)
        self.assertLess(sd, 1)
        self.assertGreater(sd, 0.4)

    def test_bootstrap_null(self):
        """Assume treatment has no effect"""
        np.random.seed(0)
        b = 250
        type1errors = 0
        for i in range(b):
            Y = np.random.binomial(1, .5, 100)
            A = np.random.binomial(1, .5, 100)

            dml = {
                'rXp': [0, 0, 0, 0] * 25,
                'mXp': [0, 0, 0, 0] * 25
            }
            actual = DML().bootstrap(Y, A, dml, 250)
            lb, ub = actual[0], actual[1]
            if not (lb <= 0 <= ub):
                type1errors += 1
        self.assertLess(type1errors, b * 0.05 + 1)
        self.assertGreater(type1errors, b * 0.01)

    def test_bootstrap_alt(self):
        """Assume treatment has effect"""
        np.random.seed(1)
        b = 250
        coverage = 0
        assertion_errors = 0
        for i in range(b):
            beta = 1
            A = np.random.binomial(1, .5, 20)
            Y = np.random.binomial(1, expit(beta * A), 20)

            dml = {
                'rXp': [0, 0, 0, 0]*5,
                'mXp': [0, 0, 0, 0]*5
            }
            try:
                lb, ub, _, _ = DML().bootstrap(Y, A, dml, 200)
                if lb <= beta <= ub:
                    coverage += 1
            except AssertionError:
                assertion_errors += 1
        self.assertLess(assertion_errors, b * .10)
        self.assertGreater(coverage, b * .8)


if __name__ == '__main__':
    unittest.main()
