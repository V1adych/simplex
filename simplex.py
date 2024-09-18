import numpy as np


def simplex(A, b, c, tol=1e-6):
    m, n = A.shape

    A_eq = np.hstack([A, np.eye(m)])
    c_eq = np.concatenate([c, np.zeros(m)])
    B = list(range(n, n + m))
    tableau = np.hstack([A_eq, b.reshape(-1, 1)])
    tableau = np.vstack([tableau, np.concatenate([c_eq, [0]])])

    def pivot_col():
        last_row = tableau[-1, :-1]
        if np.all(last_row >= -tol):
            return -1
        return np.argmin(last_row)

    def pivot_row(col):
        rhs = tableau[:-1, -1]
        lhs = tableau[:-1, col]
        ratios = np.full_like(rhs, np.inf)
        valid = lhs > tol
        ratios[valid] = rhs[valid] / lhs[valid]
        if np.all(ratios == np.inf):
            return -1
        return np.argmin(ratios)

    while True:
        col = pivot_col()
        if col == -1:
            break
        row = pivot_row(col)
        if row == -1:
            raise ValueError("The problem is unbounded.")

        tableau[row, :] /= tableau[row, col]
        for i in range(len(tableau)):
            if i != row:
                tableau[i, :] -= tableau[i, col] * tableau[row, :]

        B[row] = col

    x = np.zeros(n + m)
    x[B] = tableau[:-1, -1]

    return x[:n]




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
    b = np.array([200, 1, 0.6, 0.6, 0.6, 0.2, 0.05], dtype=np.float32)
    c = -np.array([200, 160, 260, 150, 400], dtype=np.float32)
    x = simplex(A, b, c)
    print("Optimal solution:", x)
    print("Optimal value:", -c @ x)

if __name__ == "__main__":
    main()