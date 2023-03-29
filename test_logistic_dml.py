import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from logistic_dml import *


class TestL(unittest.TestCase):
    def test_L(self):
        # Test case with binary R
        R = np.array([0, 1, 1, 0, 1])
        C = pd.DataFrame({'X1': [1, 4, 7, 10, 13], 'X2': [2, 5, 8, 11, 14], 'X3': [3, 6, 9, 12, 15]})
        Ctest = pd.DataFrame({'X1': [16, 19], 'X2': [17, 20], 'X3': [18, 21]})
        model = LogisticRegression(random_state=42)
        expected = np.array([0.85, 0.9])
        np.testing.assert_allclose(np.round(L(R, C, model, Ctest), 2), expected, rtol=1e-6)

        # Test case with continuous R
        R = np.array([0.2, 0.5, 0.8])
        C = pd.DataFrame({'X1': [1, 3, 5], 'X2': [2, 4, 6]})
        Ctest = pd.DataFrame({'X1': [7, 9], 'X2': [8, 10]})
        model = LinearRegression()
        expected = np.array([1.1, 1.4])
        np.testing.assert_allclose(np.round(L(R, C, model, Ctest), 2), expected, rtol=1e-6)


class TestLogit(unittest.TestCase):
    def test_logit(self):
        x = np.array([0.5, 0.25, 0.75])
        expected = np.array([0, -1.0986122886681098, 1.0986122886681098])
        np.testing.assert_allclose(logit(x), expected)

        x = np.array([-1])
        with np.testing.assert_raises(ValueError):
            logit(x)

        x = np.array([2])
        with np.testing.assert_raises(ValueError):
            logit(x)


if __name__ == '__main__':
    unittest.main()
