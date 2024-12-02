from scipy import sparse
from scipy.sparse.linalg import spsolve
import numpy as np
from matplotlib import pyplot as plt

class BlackScholesCNSolver:

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

        S = self.__generate_asset_grid()
        T = self.__generate_time_grid()

        dS = S[1] - S[0]
        dT = T[1] - T[0]

        alpha = 0.25 * dT * ((self.__sigma**2) * (S**2) - self.__r*S)
        beta = -dT * 0.5 * (self.__sigma**2 * (S**2) + self.__r)
        gamma = 0.25 * dT * (self.__sigma**2 * (S**2) + self.__r * S)

        lhs = sparse.diags([-alpha[2:], 1-beta[1:], -gamma[1:-1]], [-1, 0, 1], shape = (self.__s_nodes - 1, self.__s_nodes - 1), format='csr')
        rhs = sparse.diags([alpha[2:], 1+beta[1:], gamma[1:-1]], [-1, 0, 1], shape = (self.__s_nodes - 1, self.__s_nodes - 1) , format='csr')

        V = np.zeros((self.__s_nodes+1, self.__t_nodes+1))

        # setting terminal condition (for all values of S at time T)
        V[:,-1] = np.maximum(S - self.__K, 0)

        # setting boundary conditions (for all values of t at asset prices S=0 and S=Smax)
        V[0, :] = 0
        V[-1, :] = S[-1] - self.__K * np.exp(-self.__r * (self.__expiry - T))

        for tau in reversed(range(self.__t_nodes)):
            # Construct the RHS vector for this time step
            rhs_vector = rhs @ V[1:-1, tau + 1]

            # Apply boundary conditions to the RHS vector
            rhs_vector[0] += alpha[1] * (V[0, tau + 1] + V[0, tau])
            rhs_vector[-1] += gamma[self.__s_nodes-1] *(V[-1, tau+1] + V[-1, tau])

            # Solve the linear system for interior points
            V[1:-1, tau] = spsolve(lhs, rhs_vector)

        self.__V = V
        # print(self.__V)
        return self


    def __generate_asset_grid(self):
        return np.linspace(0, self.__S_max, self.__s_nodes+1)

    def __generate_time_grid(self):
        return np.linspace(0, self.__expiry, self.__t_nodes+1)

    def get_V(self):
        return self.__V

    def plot(self):

        S = self.__generate_asset_grid()
        T = self.__generate_time_grid()

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


def main():
    BlackScholesCNSolver('call', 300, 1, 0.2, 0.05, 100, 200, 100).solve().plot()

if __name__ == "__main__":
    main()