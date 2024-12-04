import numpy as np

from pdesolvers.heat_equation_1d.solution.solution_1d import Solution1D
from pdesolvers.heat_equation_1d.solver.solver_strategy import SolverStrategy


class HeatEquationExplicitSolver(SolverStrategy):

    def __init__(self, equation):
        self.equation = equation

    def solve(self):

        x = self.equation.generate_x_grid()
        dx = x[1] - x[0]

        dt_max = 0.5 * (dx**2) / self.equation.get_k()
        dt = 0.8 * dt_max
        time_step = int(self.equation.get_time()/dt)
        self.equation.set_t_nodes(time_step)

        t = np.linspace(0, self.equation.get_time(), self.equation.get_t_nodes())

        u = np.zeros((time_step, self.equation.get_x_nodes()))

        u[0, :] = self.equation.get_initial_temp(x)
        u[:, 0] = self.equation.get_left_boundary(t)
        u[:, -1] = self.equation.get_right_boundary(t)

        for tau in range(0, time_step-1):
            for i in range(1, self.equation.get_x_nodes() - 1):
                u[tau+1,i] = u[tau, i] + (dt * self.equation.get_k() * (u[tau, i-1] - 2 * u[tau, i] + u[tau, i+1]) / dx**2)

        return Solution1D(u, x, t)
