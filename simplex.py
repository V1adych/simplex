import numpy as np
from typing import Tuple


def simplex(
    A: np.ndarray, b: np.ndarray, c: np.ndarray, tol: float = 1e-6
) -> Tuple[bool, np.ndarray, float]:
    """
    Solves a linear programming problem using the simplex method.

    Args:
    A: A numpy array representing the coefficients of the constraints.
    b: A numpy array representing the right-hand side of the constraints.
    c: A numpy array representing the coefficients of the objective function.
    tol: A tolerance value for the simplex method.

    Returns:
    A tuple containing three elements:
    - A boolean value indicating whether the simplex method was successful.
    - A numpy array representing the solution to the linear programming problem.
    - The value of the objective function at the solution.
    """
    m, n = A.shape

    A_eq = np.hstack([A, np.eye(m)])
    c_eq = np.concatenate([c, np.zeros(m)])
    B = list(range(n, n + m))
    tableau = np.hstack([A_eq, b.reshape(-1, 1)])
    tableau = np.vstack([tableau, np.concatenate([c_eq, [0]])])

    while True:
        col = pivot_col(tableau, tol)
        if col == -1:
            break
        row = pivot_row(tableau, tol, col)
        if row == -1:
            return False, None, None

        tableau[row, :] /= tableau[row, col]
        for i in range(len(tableau)):
            if i != row:
                tableau[i, :] -= tableau[i, col] * tableau[row, :]

        B[row] = col

    x = np.zeros(n + m)
    x[B] = tableau[:-1, -1]

    return True, x[:n], c @ x[:n]


def pivot_col(tableau: np.ndarray, tol: float) -> int:
    last_row = tableau[-1, :-1]
    if np.all(last_row >= -tol):
        return -1
    return np.argmin(last_row)


def pivot_row(tableau: np.ndarray, tol: float, col: int) -> int:
    rhs = tableau[:-1, -1]
    lhs = tableau[:-1, col]
    ratios = np.full_like(rhs, np.inf)
    valid = lhs > tol
    ratios[valid] = rhs[valid] / lhs[valid]
    if np.all(ratios == np.inf):
        return -1
    return np.argmin(ratios)


def main():
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
    b = np.array([200, 0.01, 0.6, 0.6, 0.6, 0.2, 0.05], dtype=np.float32)
    c = -np.array([200, 160, 260, 150, 400], dtype=np.float32)
    state, x, f = simplex(A, b, c)
    print("Solver state:", "solved" if state else "not solved")
    print("Optimal solution:", x)
    print("Optimal value:", f)


if __name__ == "__main__":
    main()
