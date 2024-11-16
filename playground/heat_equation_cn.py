import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix
from matplotlib import pyplot as plt


class HeatEquationCNSolver:
    def __init__(self, length, x_nodes, time, t_nodes, k):
        self.__length = length
        self.__x_nodes = x_nodes
        self.__time = time
        self.__t_nodes = t_nodes
        self.__k = k
        self.__initial_temp = None
        self.__left_boundary_temp = None
        self.__right_boundary_temp = None
        self.__u = None

    def __check_conditions(self):
        if self.__initial_temp is None:
            return ValueError("Initial Temperature has not been initialised")

        if self.__left_boundary_temp is not None:
            err = np.abs(self.__left_boundary_temp(0) - self.__initial_temp(0))
            assert err < 1e-12

        if self.__right_boundary_temp is not None:
            err = np.abs(self.__right_boundary_temp(0) - self.__initial_temp(0))
            assert err < 1e-12


    def set_initial_temp(self, u0):
        if self.__length is None:
            return RuntimeError("Rod length has not been initialised.")
        self.__initial_temp = u0
        self.__check_conditions()
        return self

    def set_left_boundary_temp(self, left):
        self.__left_boundary_temp = left
        self.__check_conditions()
        return self

    def set_right_boundary_temp(self, right):
        self.__right_boundary_temp = right
        self.__check_conditions()
        return self

    def get_x_grid(self):
        return np.linspace(0, self.__length, self.__x_nodes)

    def get_t_grid(self):
        return np.linspace(0, self.__time, self.__t_nodes)

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


    def solve(self):
        self.__check_conditions()

        x = self.get_x_grid()
        t = self.get_t_grid()

        dx = x[1] - x[0]
        dt = t[1] - t[0]

        alpha = self.__k * dt / (2 * dx**2)
        a = -alpha
        b = 1 + 2 * alpha
        c = -alpha

        u = np.zeros((self.__t_nodes, self.__x_nodes))

        u[0, :] = self.__initial_temp(x)
        u[:, 0] = self.__left_boundary_temp(t)
        u[:, -1] = self.__right_boundary_temp(t)

        lhs = self.__build_tridiagonal_matrix(a, b, c, self.__x_nodes-2)
        rhs = np.zeros(self.__x_nodes-2)

        for tau in range(0, self.__t_nodes - 1):
            rhs[0] = alpha * (u[tau, 0] + u[tau+1, 0]) + (1 - 2 * alpha) * u[tau, 1] + alpha * u[tau, 2]

            for i in range(1, self.__x_nodes-2):
                rhs[i] = alpha * u[tau, i] + (1 - 2 * alpha) * u[tau, i+1] + alpha * u[tau, i+2]

            rhs[-1] = alpha * (u[tau, -1] + u[tau+1, -1]) + (1 - 2 * alpha) * u[tau, -2] + alpha * u[tau, -3]

            u[tau+1, 1:-1] = spsolve(lhs, rhs)

            self.__u = u

        return self

    def plot(self):

        x = self.get_x_grid()
        t = self.get_t_grid()
        x_plot, t_plot = np.meshgrid(x,t)

        # plotting the 3d surface
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(x_plot, t_plot, self.__u, cmap='viridis')

        # set labels and title
        ax.set_xlabel('Space')
        ax.set_ylabel('Time')
        ax.set_zlabel('Temperature')
        ax.set_title('3D Surface Plot of 1D Heat Equation')

        plt.show()


def main():
    solver = (((HeatEquationCNSolver(1, 100,30,10000, 0.01)
                .set_initial_temp(lambda x: np.sin(np.pi * x) + 5))
               .set_left_boundary_temp(lambda t: 20 * np.sin(np.pi * t) + 5))
              .set_right_boundary_temp(lambda t: t + 5))

    solver.solve().plot()


if __name__ == "__main__":
    main()

