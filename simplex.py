import numpy as np

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

b = np.array([300, 0.6, 0.6, 0.5, 0.2, 0.1], dtype=np.float32)
c = np.array([200, 160, 260, 150, 400], dtype=np.float32)


def is_solved(tabletau: np.array, m: int, n: int) -> bool:
    return np.all(tabletau[0, 1 : m + n + 1] > -1e-4)


def eq_zero(x):
    return abs(x - 0) < 1e-4


def simplex(A: np.array, b: np.array, c: np.array):
    m, n = A.shape
    A = A.copy().astype(np.float32)
    b = b.copy().astype(np.float32)
    c = c.copy().astype(np.float32)

    tabletau = A.copy()
    tabletau = np.hstack([tabletau, np.eye(m)])
    tabletau = np.hstack([tabletau, b.reshape(-1, 1)])
    z = np.zeros(shape=(m, 1))
    tabletau = np.hstack([z, tabletau])
    c = np.concatenate(
        [np.array([1.0]), -c, np.zeros(shape=(m + 1,), dtype=np.float32)]
    )
    tabletau = np.vstack([c, tabletau])
    basis = np.arange(n, n + m)
    i = 0
    while not is_solved(tabletau, m, n):
        for col in range(1, tabletau.shape[1] - 1):
            if tabletau[0, col] < -1e-4:
                break

        limiting_row = -1
        limiting_value = float("inf")
        for row in range(1, tabletau.shape[0]):
            if tabletau[row, -1] < 1e-4:
                continue
            if eq_zero(tabletau[row, col]):
                continue
            ratio = tabletau[row, -1] / tabletau[row, col]

            if ratio < limiting_value:
                limiting_value = ratio
                limiting_row = row

        basis[limiting_row - 1] = col - 1
        if limiting_row == -1:
            return None

        for row in range(tabletau.shape[0]):
            if row == limiting_row:
                continue
            if eq_zero(tabletau[row, col]):
                continue
            ratio = tabletau[row, col] / tabletau[limiting_row, col]
            tabletau[row] -= ratio * tabletau[limiting_row]

        i += 1
        if i >= m * n:
            return None

    target = tabletau[0, -1]
    x = np.zeros(shape=(n,), dtype=np.float32)
    for i, idx in enumerate(basis):
        if idx < n:
            x[idx] = tabletau[i + 1, -1]

    return x, target


x, target = simplex(A, b, c)
print(x)
print(target)
