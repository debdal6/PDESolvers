import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class BlackScholesExplicitSolver():

    # using call option as an example for now - will update it to toggle between put/call

    def __init__(self, S_max, expiry, sigma, r, K, s_nodes, t_nodes = None):
        self.__S_max = S_max
        self.__expiry = expiry
        self.__sigma = sigma
        self.__r = r
        self.__K = K
        self.__s_nodes = s_nodes
        self.__t_nodes = t_nodes
        self.__V = None

    def solve(self):

        if self.__t_nodes is None:
            dt_max = 1/((self.__s_nodes**2) * (self.__sigma**2)) # cfl condition to ensure stability
            dt = 0.9 * dt_max
            self.__t_nodes = int(self.__expiry/dt)
            dt = self.__expiry / self.__t_nodes # to ensure that the expiration time is integer time steps away
        else:
            # possible fix - set a check to see that user-defined value is within cfl condition
            dt = self.__expiry/self.__t_nodes

        S = self.__generate_asset_grid()
        T = self.__generate_time_grid()

        dS = S[1] - S[0]

        V = np.zeros((self.__s_nodes+1, self.__t_nodes+1))

        V[:,-1] = np.maximum(S - self.__K, 0)

        for tau in reversed(range(self.__t_nodes)):
            for i in range(1, self.__s_nodes):
                delta = (V[i+1, tau+1] - V[i-1, tau+1]) /  (2 * dS)
                gamma = (V[i+1, tau+1] - 2 * V[i,tau+1] + V[i-1, tau+1]) / dS ** 2
                theta = -0.5 * ( self.__sigma ** 2) * (S[i] ** 2) * gamma - self.__r * S[i] * delta + self.__r * V[i, tau+1]
                V[i, tau] = V[i, tau + 1] - (theta * dt)

            V[0, tau] = 0
            V[self.__s_nodes, tau] = self.__S_max - self.__K * np.exp(-self.__r * (self.__expiry - T[tau]))

        self.__V = V

        return self

    def plot(self):

        S = self.__generate_asset_grid()
        T = self.__generate_time_grid()

        X, Y = np.meshgrid(T, S)

        # plotting the 3d surface
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, self.__get_V(), cmap='viridis')

        ax.set_xlabel('Time')
        ax.set_ylabel('Asset Price')
        ax.set_zlabel('Option Value')
        ax.set_title('Option Value Surface Plot')

        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()


    def __generate_asset_grid(self):
        return np.linspace(0, self.__S_max, self.__s_nodes + 1)

    def __generate_time_grid(self):
        return np.linspace(0, self.__expiry, self.__t_nodes + 1)

    def __get_V(self):
        return self.__V

def main():
    BlackScholesExplicitSolver(300, 1, 0.2, 0.05, 100, 20).solve().plot()

if __name__ == "__main__":
    main()

