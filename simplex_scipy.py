import numpy as np
from scipy.optimize import linprog


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
solution = linprog(A_ub=A, b_ub=b, c=c)

print(solution.x)
print(solution.fun)
