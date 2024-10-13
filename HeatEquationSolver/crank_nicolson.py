import numpy as np
# from matplotlib import pyplot

"""

The code below attempts to solve the 1D Heat Equation
using the implicit Crank Nicolson method

"""

# defining the parameters

length = 2     # length of the metal rod
k = 1          # using a heat constant of 1
temp_left = 200
temp_right = 200

time = 10

dx = 0.1
x_grid = np.linspace(0, length, int(length/dx))

dt = 0.0001
t_grid = np.linspace(0, time, int(time/dt))

