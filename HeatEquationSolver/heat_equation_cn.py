import numpy as np
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# setting up parameters
length = 1
nodes = 100
time = 30
k = 0.01
initial_temp = 25


def left_temp_func(temp, t):
    """
    Generates the temperature at the left boundary of the rod

    :param temp: initial temperature of the rod at t=0
    :param t: array consisting of time steps from t=0 to t= total time
    :return: the temperature at the left boundary
    """
    return 20 * np.sin(np.pi * t) + temp


def right_temp_func(temp, t):
    """
    Generates the temperature at the right boundary of the rod

    :param temp: initial temperature of the rod at t=0
    :param t: array consisting of time steps from t=0 to t= total time
    :return: the temperature at the right boundary
    """
    return t + temp


def build_tridiagonal_matrix(a, b, c, nodes):
    """
    Initialises the tridiagonal matrix on the LHS of the equation

    :param a: the coefficient of U @ (t = tau + 1 & x = i-1)
    :param b: the coefficient of U @ (t = tau + 1 & x = i)
    :param c: the coefficient of U @ (t = tau + 1 & x = i+1)
    :param nodes: number of spatial nodes ( used to initialise the size of the tridiagonal matrix)
    :return: the tridiagonal matrix consisting of coefficients
    """

    matrix = np.zeros((nodes, nodes))
    np.fill_diagonal(matrix, b)
    np.fill_diagonal(matrix[1:], a)
    np.fill_diagonal(matrix[:, 1:], c)

    return matrix


def solve_heat_equation_cn(length, nodes, time, k, initial_temp):
    """
    Solves the heat equation using the crank nicolson method

    :param length: length of the rod
    :param nodes: number of spatial nodes across the rod
    :param time: the total time
    :param k: heat diffusivity constant
    :param initial_temp: initial temperature of the rod at t=0
    :return: - u: the solved 2d matrix consisting the temperature distribution over time
             - dt: time step size (used for plotting)
             - time_step: no. of time steps for the simulation (used for plotting)
    """
    dx = length / (nodes-1)
    dt_max = 0.5 * (dx**2) / k  # calculating dt using a formula (to ensure stability)
    dt = 0.8 * dt_max
    time_step = int(time/dt)
    t = np.linspace(0, time, time_step)

    alpha = k * dt / (2 * dx**2)
    a = -alpha
    b = 1 + 2 * alpha
    c = -alpha

    u = np.zeros((time_step, nodes))

    u[0, :] = initial_temp
    u[:, 0] = left_temp_func(initial_temp, t)
    u[:, -1] = right_temp_func(initial_temp, t)

    lhs = build_tridiagonal_matrix(a, b, c, nodes-2)
    rhs = np.zeros(nodes-2)

    for tau in range(0, time_step - 1):
        rhs[0] = alpha * (u[tau, 0] + u[tau+1, 0]) + (1 - 2 * alpha) * u[tau, 1] + alpha * u[tau, 2]

        for i in range(1, nodes-2):
            rhs[i] = alpha * u[tau, i] + (1 - 2 * alpha) * u[tau, i+1] + alpha * u[tau, i+2]

        rhs[-1] = alpha * (u[tau, -1] + u[tau+1, -1]) + (1 - 2 * alpha) * u[tau, -2] + alpha * u[tau, -3]

        u[tau+1, 1:-1] = spsolve(lhs, rhs)

    return u, dt, time_step


def plot(u, dt, time_step):
    """
    Plots the temperature distribution of the rod over time

    :param u: the solved 2d matrix consisting the temperature distribution over time
    :param dt: time step size
    :param time_step: no of time steps for the simulation
    :return: an animated diagram of temperature distribution
    """
    # setting up plots
    fig, ax = plt.subplots()
    ax.set_title('Heat Distribution over time')
    ax.set_xlabel('Length across the rod')
    ax.set_ylabel('Temperature')
    # plotting initial temperature distribution at t = 0
    x = np.linspace(0, length, nodes)
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
    u, dt, time_step = solve_heat_equation_cn(length, nodes, time, k, initial_temp)
    print(u)
    plot(u, dt, time_step)


if __name__ == "__main__":
    main()

