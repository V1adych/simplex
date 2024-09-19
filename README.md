# Linear Programming Solver

This repository contains Python implementations of linear programming solvers using the `SciPy` library and a custom Simplex method. The goal is to maximize the nutritious value of a salad while meeting constraints on the cost of its ingredients, fat content, and ingredient weights.

## Project Overview

We aim to solve a linear programming problem with the following constraints:
- **Ingredients:** Tomato, Cucumber, Bell Pepper, Lettuce Leaf, Onion.
- **Objective:** Maximize the nutritious value of the salad.
- **Constraints:** Cost, fat concentration, and maximum weight of each ingredient.

The problem is solved using two methods:
1. **SciPy's `linprog` function**.
2. **Custom Simplex method implementation**.

### File hierarchy
```bash
├── README.md
├── LICENSE
├── tests/
│   └── test_simplex.py # tests for comparing results between our method and scipy
├── .gitignore
├── simplex.py # our implementation of simplex
└── simplex_scipy.py # scipy implementation of simplex
```

## Problem Setup
- **Problem statement:**  
   Maximize $c^\top x$  with constraint $Ax \leq b$ where $x$ is the amount of each ingredient, $c$ is the vector of the nutritious value per kilogram, $A$ is constraint coefficients, and $b$ is constraint upper bounds.

- **Objective function:**  
   Maximize nutritious value of the salad measured in kilocalories (kcal).
  
- **Constraints:**  
   The system is constrained by the maximum allowed weights of ingredients (kg), their costs (rub), and fat content proportions (kg).

### Input Matrix

- **Cost Matrix (A):**

| Ingredient    | Cost (rub/kg) | Fat Proportion | Max Weight (kg) |
| ------------- | ------------- | ---------------| --------------- |
| Tomato        | 130           | 0.004          | 0.6             |
| Cucumber      | 100           | 0.005          | 0.6             |
| Bell Pepper   | 155           | 0.006          | 0.6             |
| Lettuce Leaf  | 85            | 0.003          | 0.2             |
| Onion         | 50            | 0.004          | 0.05            |

### Constraints
The system aims to satisfy the following:
1. The total cost of the ingredients should not exceed 200 rubles.
2. The total fat content should be below 10 grams.
3. Each ingredient has a maximum allowable weight (to avoid cooking Onion Frenzy).

## Dependencies

- Python 3.x
- NumPy
- SciPy
- PyTest (optional) 

You can install the required dependencies using pip:

```bash
pip install numpy scipy pytest
```

## Usage

To run tests for comparing `scipy` implementation with ours, run:
```bash
pytest
```
To get result of our implementation of simplex method:
```bash
python simplex.py
```
To view `scipy` results:
```bash
python simplex_scipy.py
```

## Results

Both our implementation and `scipy` show same optimal solution of the LP problem:

- **Optimal solution (x)**

| Ingredient    | Resulting x (kg)|
| ------------- | --------------- |
| Tomato        | 0.2115          |
| Cucumber      | 0.6             |
| Bell Pepper   | 0.6             |
| Lettuce Leaf  | 0.2             |
| Onion         | 0.05            |

- **Target function $(c^\top x)$**

| Optimal value (kcal) |
| -------------------- |
| 344.4                |

Congratulations! We have found most nutritious salad receipt for 200 rub budget!
## Authors

- Nikita Zagainov, Ilyas Galiev, Arthur Babkin, Nikita Menshikov
