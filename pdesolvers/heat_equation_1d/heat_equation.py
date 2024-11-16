import numpy as np
from matplotlib import pyplot as plt

from pdesolvers.heat_equation_1d.strategy.solver_strategy import SolverStrategy


class HeatEquation:

    def __init__(self, length, x_nodes, time, t_nodes, k, solver: SolverStrategy = None):
        self.__length = length
        self.__x_nodes = x_nodes
        self.__time = time
        self.__t_nodes = t_nodes
        self.__k = k
        self.__solver = solver
        self.__initial_temp = None
        self.__left_boundary_temp = None
        self.__right_boundary_temp = None
        self.__u = None

    def set_solver(self, strategy: SolverStrategy):
        self.__solver = strategy

    def solve(self):
        if self.__solver is None:
            raise RuntimeError("Solver has not been set. Please pick one")

        self.__solver.solve(self)
        return self

    def plot(self):

        if self.__u is None:
            raise RuntimeError("Solution has not been computed - please run the solver.")

        x = self._generate_x_grid()
        t = self._generate_t_grid()
        x_plot, t_plot = np.meshgrid(x,t)

        # plotting the 3d surface
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(x_plot, t_plot, self._get_result(), cmap='viridis')

        # set labels and title
        ax.set_xlabel('Space')
        ax.set_ylabel('Time')
        ax.set_zlabel('Temperature')
        ax.set_title('3D Surface Plot of 1D Heat Equation')

        plt.show()

    def set_initial_temp(self, u0):
        self.__validate_callable(u0)
        if self.__length is None:
            raise RuntimeError("Rod length has not been initialised.")
        self.__initial_temp = u0
        self.__check_conditions()
        return self

    def set_left_boundary_temp(self, left):
        self.__validate_callable(left)
        self.__left_boundary_temp = left
        self.__check_conditions()
        return self

    def set_right_boundary_temp(self, right):
        self.__validate_callable(right)
        self.__right_boundary_temp = right
        self.__check_conditions()
        return self

    def __check_conditions(self):
        if self.__initial_temp is None:
            raise ValueError("Initial Temperature has not been initialised")

        if self.__left_boundary_temp is not None:
            err = np.abs(self.__left_boundary_temp(0) - self.__initial_temp(0))
            assert err < 1e-12

        if self.__right_boundary_temp is not None:
            err = np.abs(self.__right_boundary_temp(0) - self.__initial_temp(0))
            assert err < 1e-12

    @staticmethod
    def __validate_callable(func):
        if not callable(func):
            raise ValueError("Temperature conditions must be a callable function")

    def _generate_x_grid(self):
        return np.linspace(0, self.__length, self.__x_nodes)

    def _generate_t_grid(self):
        return np.linspace(0, self.__time, self.__t_nodes)

    def _set_result(self, u):
        self.__u = u
        return self

    def _set_t_nodes(self, nodes):
        self.__t_nodes = nodes

    def _get_length(self):
        return self.__length

    def _get_time(self):
        return self.__time

    def _get_x_nodes(self):
        return self.__x_nodes

    def _get_t_nodes(self):
        return self.__t_nodes

    def _get_k(self):
        return self.__k

    def _get_initial_temp(self, x):
        return self.__initial_temp(x)

    def _get_left_boundary(self, t):
        return self.__left_boundary_temp(t)

    def _get_right_boundary(self, t):
        return self.__right_boundary_temp(t)

    def _get_result(self):
        return self.__u

