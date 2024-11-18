import numpy as np
from matplotlib import pyplot as plt

from pdesolvers.heat_equation_1d.heat_equation import HeatEquation
from pdesolvers.heat_equation_1d.solver.solver_strategy import SolverStrategy


class HeatEquationExplicitSolver(SolverStrategy):


    def solve(self, equation : HeatEquation):

        self.equation = equation

        x = self.equation._generate_x_grid()
        dx = x[1] - x[0]

        dt_max = 0.5 * (dx**2) / self.equation._get_k()
        dt = 0.8 * dt_max
        time_step = int(self.equation._get_time()/dt)
        self.equation._set_t_nodes(time_step)

        t = np.linspace(0, self.equation._get_time(), self.equation._get_t_nodes())

        u = np.zeros((time_step, self.equation._get_x_nodes()))

        u[0, :] = self.equation._get_initial_temp(x)
        u[:, 0] = self.equation._get_left_boundary(t)
        u[:, -1] = self.equation._get_right_boundary(t)

        for tau in range(0, time_step-1):
            for i in range(1, self.equation._get_x_nodes() - 1):
                u[tau+1,i] = u[tau, i] + (dt * self.equation._get_k() * (u[tau, i-1] - 2 * u[tau, i] + u[tau, i+1]) / dx**2)

        self.equation._set_result(u)

        return self

    def plot(self):

        if self.equation._get_result() is None:
            raise RuntimeError("Solution has not been computed - please run the solver.")

        x = self.equation._generate_x_grid()
        t = self.equation._generate_t_grid()
        x_plot, t_plot = np.meshgrid(x,t)

        # plotting the 3d surface
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(x_plot, t_plot, self.equation._get_result(), cmap='viridis')

        # set labels and title
        ax.set_xlabel('Space')
        ax.set_ylabel('Time')
        ax.set_zlabel('Temperature')
        ax.set_title('3D Surface Plot of 1D Heat Equation')

        plt.show()