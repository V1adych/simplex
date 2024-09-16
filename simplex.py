import numpy as np


def simplex(A, b, c):
    m, n = A.shape

    tableau = np.zeros((m + 1, n + m + 1))
    tableau[:-1, :-1] = np.hstack((A, np.eye(m)))
    tableau[:-1, -1] = b
    tableau[-1, :-1] = np.hstack((-c, np.zeros(m)))

    def pivot(tableau, row, col):
        tableau[row, :] /= tableau[row, col]
        for r in range(tableau.shape[0]):
            if r != row:
                tableau[r, :] -= tableau[r, col] * tableau[row, :]

    while True:
        if all(tableau[-1, :-1] >= 0):
            break

        col = np.argmin(tableau[-1, :-1])

        if all(tableau[:-1, col] <= 0):
            return None

        ratios = tableau[:-1, -1] / tableau[:-1, col]
        valid_ratios = [(i, ratio) for i, ratio in enumerate(ratios) if ratio > 0]
        row, _ = min(valid_ratios, key=lambda x: x[1])

        pivot(tableau, row, col)

    x = np.zeros(n)
    for i in range(m):
        if np.argmax(tableau[i, :n]) < n:
            x[np.argmax(tableau[i, :n])] = tableau[i, -1]

    return x


A = np.array(
    [
        [120, 100, 150, 80, 50],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ],
    dtype=np.float32,
)

b = np.array([200, 0.6, 0.6, 0.6, 0.2, 0.05], dtype=np.float32)
c = np.array([200, 160, 260, 150, 400], dtype=np.float32)

solution = simplex(A, b, c)
print("Optimal solution:", solution)
