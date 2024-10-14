import matplotlib.pyplot as plt
import numpy as np

"""

The code below attempts to solve the 1D Heat Equation
using the explicit scheme

"""

# defining the parameters

L = 50      # length of rod in mm
t = 4       # time in seconds
x_nodes = 10
k = 110  # diffusivity constant


dx = L/x_nodes
dt = 0.5 * (dx**2) / k   # calculating dt using a formula (to ensure stability)
t_nodes = int(t/dt)

# defining an initial vector (matrix)
u = np.zeros(x_nodes) + 20  # sets the rod to have an initial temp of 20 deg

# boundary conditions
u[0] = 100      # setting first element to 100
u[-1] = 100     # setting last element to 100

# visualization

# creates a figure and a set of axes
fig, axis = plt.subplots()

# creates a colour mesh
pcm = axis.pcolormesh(np.array([u]), cmap='jet', vmin=0, vmax=100)

# colour bar for visual colour reference
plt.colorbar(pcm, ax=axis)

# sets the y-axis limits
axis.set_ylim([-2, 3])

# simulation

counter = 0

while counter < t:
    w = u.copy()

    # at every node - calculate the temp
    for i in range(1, x_nodes - 1):
        u[i] = w[i] + (dt * k * (w[i-1] - 2 * w[i] + w[i+1]) / dx**2)

    counter += dt

    print("t: {:.3f} [s], Average temperature: {:.2f} Celsius".format(counter, np.average(u)))

    # updating the plot w vector u
    pcm.set_array([u])
    axis.set_title("Distribution at t: {:.3f} [s].".format(counter))
    plt.pause(0.01)         # updates plot every 10 milliseconds

plt.show()
