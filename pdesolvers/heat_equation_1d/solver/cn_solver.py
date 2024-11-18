import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix

from pdesolvers.heat_equation_1d.heat_equation import HeatEquation
from pdesolvers.heat_equation_1d.solver.solver_strategy import SolverStrategy


class HeatEquationCNSolver(SolverStrategy):

    def solve(self, equation : HeatEquation):

        self.equation = equation

        x = self.equation._generate_x_grid()
        t = self.equation._generate_t_grid()

        dx = x[1] - x[0]
        dt = t[1] - t[0]

        alpha = self.equation._get_k() * dt / (2 * dx**2)
        a = -alpha
        b = 1 + 2 * alpha
        c = -alpha

        u = np.zeros((self.equation._get_t_nodes(), self.equation._get_x_nodes()))

        u[0, :] = self.equation._get_initial_temp(x)
        u[:, 0] = self.equation._get_left_boundary(t)
        u[:, -1] = self.equation._get_right_boundary(t)

        lhs = self.__build_tridiagonal_matrix(a, b, c, self.equation._get_x_nodes() - 2)
        rhs = np.zeros(self.equation._get_x_nodes() - 2)

        for tau in range(0, self.equation._get_t_nodes() - 1):
            rhs[0] = alpha * (u[tau, 0] + u[tau+1, 0]) + (1 - 2 * alpha) * u[tau, 1] + alpha * u[tau, 2]

            for i in range(1, self.equation._get_x_nodes() - 2):
                rhs[i] = alpha * u[tau, i] + (1 - 2 * alpha) * u[tau, i+1] + alpha * u[tau, i+2]

            rhs[-1] = alpha * (u[tau, -1] + u[tau+1, -1]) + (1 - 2 * alpha) * u[tau, -2] + alpha * u[tau, -3]

            u[tau+1, 1:-1] = spsolve(lhs, rhs)

            self.equation._set_result(u)

        return self

    @staticmethod
    def __build_tridiagonal_matrix(a, b, c, nodes):
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

        matrix = csc_matrix(matrix)

        return matrix

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