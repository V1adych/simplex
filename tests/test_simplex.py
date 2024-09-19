import os
import sys
from scipy.optimize import linprog
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simplex import simplex


def test_simplex_base():
    A = np.array(
        [
            [130, 100, 155, 85, 50],
            [0.004, 0.005, 0.006, 0.003, 0.004],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    b = np.array([200, 1, 0.6, 0.6, 0.6, 0.2, 0.05], dtype=np.float32)
    c = -np.array([200, 160, 260, 150, 400], dtype=np.float32)

    _, x_result, _ = simplex(A, b, c)
    x_test = linprog(c, A, b).x

    np.testing.assert_array_almost_equal(x_result, x_test)


def test_simplex_custom_1():
    A = np.array(
        [
            [100, 500, 200, 60, 75],
            [4, 5, 6, 3, 4],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    b = np.array([130, 4, 6, 6, 6, 2, 0.5], dtype=np.float32)
    c = -np.array([130, 200, 250, 100, 430], dtype=np.float32)

    _, x_result, _ = simplex(A, b, c)
    x_test = linprog(c, A, b).x

    np.testing.assert_array_almost_equal(x_result, x_test)


def test_simplex_custom_2():
    A = np.array(
        [
            [10, 1000, 150, 90, 34],
            [0.4, 0.05, 6, 3, 4],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    b = np.array([200, 400, 600, 300, 250, 214, 5], dtype=np.float32)
    c = -np.array([201, 152, 245, 112, 321], dtype=np.float32)

    _, x_result, _ = simplex(A, b, c)
    x_test = linprog(c, A, b).x

    np.testing.assert_array_almost_equal(x_result, x_test)
