import numpy as np
from scipy.optimize import linprog

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
c = -np.array([200, 160, 260, 150, 400], dtype=np.float32)
solution = linprog(A_ub=A, b_ub=b, c=c)

print(solution.x)
print(solution.fun)
