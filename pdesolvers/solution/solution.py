import pdesolvers.pdes.black_scholes as bse
import numpy as np
from matplotlib import pyplot as plt

class Solution1D:

    def __init__(self, result, x_grid, t_grid):
        self.result = result
        self.x_grid = x_grid
        self.t_grid = t_grid

    def plot(self):
        """
        Generates a 3D surface plot of the temperature distribution across a grid of space and time

        :return: 3D surface plot
        """

        if self.result is None:
            raise RuntimeError("Solution has not been computed - please run the solver.")

        x_plot, t_plot = np.meshgrid(self.x_grid,self.t_grid)

        # plotting the 3d surface
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(x_plot, t_plot, self.result, cmap='viridis')

        # set labels and title
        ax.set_xlabel('Space')
        ax.set_ylabel('Time')
        ax.set_zlabel('Temperature')
        ax.set_title('3D Surface Plot of 1D Heat Equation')

        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    def get_result(self):
        """
        Gets the grid of computed temperature values

        :return: grid result
        """
        return self.result

class SolutionBlackScholes:
    def __init__(self, result, s_grid, t_grid, delta, gamma, theta):
        self.result = result
        self.s_grid = s_grid
        self.t_grid = t_grid
        self.delta = delta
        self.gamma = gamma
        self.theta = theta

    def plot(self):
        """
        Generates a 3D surface plot of the option values across a grid of asset prices and time

        :return: 3D surface plot
        """

        X, Y = np.meshgrid(self.t_grid, self.s_grid)

        # plotting the 3d surface
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, self.result, cmap='viridis')

        ax.set_xlabel('Time')
        ax.set_ylabel('Asset Price')
        ax.set_zlabel('Option Value')
        ax.set_title('Option Value Surface Plot')

        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    def get_result(self):
        """
        Gets the grid of computed option prices

        :return: grid result
        """
        return self.result

    def plot_greek(self, greek_type='delta', time_step=0):

        greek_types = {
            'delta': {'data': self.delta, 'title': 'Delta'},
            'gamma': {'data': self.gamma, 'title': 'Gamma'},
            'theta': {'data': self.theta, 'title': 'Theta'}
        }

        if greek_type.lower() not in greek_types:
            raise ValueError("Invalid greek type - please choose between delta/gamma/theta.")

        chosen_greek = greek_types[greek_type.lower()]
        greek_data = chosen_greek['data'][:, time_step]
        plt.figure(figsize=(8, 6))
        plt.plot(self.s_grid, greek_data, label=f"Delta at t={self.t_grid[time_step]:.4f}", color="blue")

        plt.title(f"{chosen_greek['title']} vs. Stock Price at t={self.t_grid[time_step]:.4f}")
        plt.xlabel("Stock Price (S)")
        plt.ylabel(chosen_greek['title'])
        plt.grid()
        plt.legend()

        plt.show()

