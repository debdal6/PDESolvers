# PDE Solvers

This repository contains a collection of numerical solvers for solving Partial Differential Equations (PDEs). As of now, it covers the following equations:

- **1D Heat Equation**
- **Black-Scholes Equation**

Additional features include:
- **Geometric brownian motion** : used to simulate multiple price paths, given the current price of an asset
  
It includes two key components:

- **Python Library**: A general-purpose solver for PDEs implemented using numerical methods.
- **CUDA Library**: A GPU-accelerated version of the solvers for faster and efficient computations.

Numerical Methods used to solve partial differential equations:
- **Explicit Method**
- **Crank-Nicolson Method**

## Requirements

### Python Library

- NumPy
- SciPy
- Matplotlib for visualizations

Install the required python packages:
```bash
pip install -r requirements.txt
