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

### Problem Setup

- **Objective function:**  
   Maximize `c.T @ x`  
   where `x` is the amount of each ingredient, and `c` is the vector of the nutritious value per kilogram.
  
- **Constraints:**  
   The system is constrained by the maximum allowed weights of ingredients, their costs, and fat content proportions.

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
2. The total fat content should be below a specified limit.
3. Each ingredient has a maximum allowable weight.

## Dependencies

- Python 3.x
- NumPy
- SciPy

You can install the required dependencies using pip:

```bash
pip install numpy scipy
```

## Usage

### SciPy Linear Programming Solver

The first approach utilizes `SciPy`'s built-in `linprog` function for solving the linear programming problem.

## Results

The method outputs the optimal quantities of ingredients that maximize the nutritious value while satisfying the constraints. For example, with the given parameters:

## Authors

- Nikita Zagainov, Ilyas Galiev, Arthur Babkin, Nikita Menshikov
