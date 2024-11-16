import numpy as np

from pdesolvers.heat_equation_1d.strategy.solver_strategy import SolverStrategy


class HeatEquationExplicitSolver(SolverStrategy):

    def solve(self, context):

        x = context._generate_x_grid()
        dx = x[1] - x[0]

        dt_max = 0.5 * (dx**2) / context._get_k()
        dt = 0.8 * dt_max
        time_step = int(context._get_time()/dt)
        context._set_t_nodes(time_step)

        t = np.linspace(0, context._get_time(), context._get_t_nodes())

        u = np.zeros((time_step, context._get_x_nodes()))

        u[0, :] = context._get_initial_temp(x)
        u[:, 0] = context._get_left_boundary(t)
        u[:, -1] = context._get_right_boundary(t)

        for tau in range(0, time_step-1):
            for i in range(1, context._get_x_nodes() - 1):
                u[tau+1,i] = u[tau, i] + (dt * context._get_k() * (u[tau, i-1] - 2 * u[tau, i] + u[tau, i+1]) / dx**2)

        context._set_result(u)

        return self