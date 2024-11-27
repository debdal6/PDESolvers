import numpy as np
from matplotlib import pyplot as plt


class BlackScholesExplicitSolver:

    """
    The Black Scholes equation is a Partial Differential Equation (PDE) used to determine the fair value price of an
    option over time.

    The solver takes into account two different option types - call and put options
    """

    def __init__(self, option_type, S_max, expiry, sigma, r, K, s_nodes, t_nodes = None):
        """
        Initialises the solver with the necessary parameters

        :param option_type: the type of option
        :param S_max: maximum asset price in the grid
        :param expiry: time to maturity/expiry of the option
        :param sigma: volatility of the asset
        :param r: risk-free interest rate
        :param K: strike price
        :param s_nodes: number of asset price nodes
        :param t_nodes: number of time nodes
        """

        self.__option_type = option_type
        self.__S_max = S_max
        self.__expiry = expiry
        self.__sigma = sigma
        self.__r = r
        self.__K = K
        self.__s_nodes = s_nodes
        self.__t_nodes = t_nodes
        self.__V = None

    def solve(self):
        """
        This method solves the Black-Scholes equation using the explicit finite difference method
        :return: the solver instance with the computed option values
        """

        if self.__t_nodes is None:
            dt_max = 1/((self.__s_nodes**2) * (self.__sigma**2)) # cfl condition to ensure stability
            dt = 0.9 * dt_max
            self.__t_nodes = int(self.__expiry/dt)
            dt = self.__expiry / self.__t_nodes # to ensure that the expiration time is integer time steps away
        else:
            # possible fix - set a check to see that user-defined value is within cfl condition
            dt = self.__expiry/ self.__t_nodes

        S = self.__generate_asset_grid()
        T = self.__generate_time_grid()

        dS = S[1] - S[0]

        V = np.zeros((self.__s_nodes+1, self.__t_nodes+1))

        # setting terminal condition
        if self.__option_type == 'call':
            V[:,-1] = np.maximum((S - self.__K), 0)
        elif self.__option_type == 'put':
            V[:,-1] = np.maximum((self.__K - S), 0)

        for tau in reversed(range(self.__t_nodes)):
            for i in range(1, self.__s_nodes):
                delta = (V[i+1, tau+1] - V[i-1, tau+1]) /  (2 * dS)
                gamma = (V[i+1, tau+1] - 2 * V[i,tau+1] + V[i-1, tau+1]) / dS ** 2
                theta = -0.5 * ( self.__sigma ** 2) * (S[i] ** 2) * gamma - self.__r * S[i] * delta + self.__r * V[i, tau+1]
                V[i, tau] = V[i, tau + 1] - (theta * dt)

            # setting boundary conditions
            lower, upper = self.__set_boundary_conditions(T, tau)
            V[0, tau] = lower
            V[self.__s_nodes, tau] = upper

        self.__V = V

        print(self.__V)

        return self

    def plot(self):

        S = self.__generate_asset_grid()
        T = self.__generate_time_grid()

        T = T[::-1]

        X, Y = np.meshgrid(T, S)

        # plotting the 3d surface
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, self.get_V(), cmap='viridis')

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

    def __set_boundary_conditions(self, T, tau):
        if self.__option_type == 'call':
            lower_boundary = 0
            upper_boundary = self.__S_max - self.__K * np.exp(-self.__r * (self.__expiry - T[tau]))
        elif self.__option_type == 'put':
            lower_boundary = self.__K * np.exp(-self.__r * (self.__expiry - T[tau]))
            upper_boundary = 0

        return lower_boundary, upper_boundary

    def get_V(self):
        return self.__V

def main():
    BlackScholesExplicitSolver('call', 300, 1, 0.2, 0.05, 100, 100).solve().plot()

if __name__ == "__main__":
    main()

