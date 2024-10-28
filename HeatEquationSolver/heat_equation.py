import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# setting up parameters
length = 1
nodes = 100
time = 30
k = 0.01        # diffusivity constant


def initialise_grid(length, nodes, time, k):
    dx = length / (nodes-1)
    dt_max = 0.5 * (dx**2) / k  # calculating dt using a formula (to ensure stability)
    dt = 0.8 * dt_max
    time_step = int(time/dt)
    x = np.linspace(0, length, nodes)
    t = np.linspace(0, time, time_step)

    return dx, dt, time_step, x, t


def setting_conditions(t, time_step, nodes):
    # initial and boundary conditions
    u_initial = 25
    u_left = 20 * np.sin(np.pi * t) + 25
    u_right = t + 25
    u = np.zeros((time_step, nodes))
    u[0, :] = u_initial

    return u, u_left, u_right


# initializing 2d-matrix to store temperature
def solve_heat_equation(u, u_left, u_right, time_step, nodes, dt, k, dx):
    for tau in range(0, time_step-1):
        u[tau+1, 0] = u_left[tau+1]
        u[tau+1, -1] = u_right[tau+1]
        for i in range(1, nodes - 1):
            u[tau+1,i] = u[tau, i] + (dt * k * (u[tau, i-1] - 2 * u[tau, i] + u[tau, i+1]) / dx**2)
    return u


def plot(x, u, dt, time_step,):
    # setting up plots
    fig, ax = plt.subplots()
    ax.set_title('Heat Distribution over time')
    ax.set_xlabel('Length across the rod')
    ax.set_ylabel('Temperature')
    # plotting initial temperature distribution at t = 0
    line, = ax.plot(x, u[0])
    # setting y-axis limits
    ax.set_ylim(bottom=0, top=60)

    def animate(frame):
        line.set_ydata(u[frame])
        ax.set_title(f'Heat Distribution at time = {frame * dt:.2f}s')
        return line,

    ani = FuncAnimation(fig, animate, frames=range(0, time_step, time_step // 100), blit=False)
    plt.show()


def main():
    dx, dt, time_step, x, t = initialise_grid(length, nodes, time, k)
    u, u_left, u_right = setting_conditions(t, time_step, nodes)
    u = solve_heat_equation(u, u_left, u_right, time_step, nodes, dt, k, dx)
    print(u)
    plot(x, u, dt, time_step)


if __name__ == "__main__":
    main()