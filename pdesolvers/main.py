import numpy as np
import pandas as pd
import pdesolvers as pde

def main():

    # testing for heat equation

    # equation1 = (pde.HeatEquation(1, 100,30,10000, 0.01)
    #             .set_initial_temp(lambda x: np.sin(np.pi * x) + 5)
    #             .set_left_boundary_temp(lambda t: 20 * np.sin(np.pi * t) + 5)
    #             .set_right_boundary_temp(lambda t: t + 5))
    #
    #
    # solver1 = pde.Heat1DCNSolver(equation1)
    # solver2 = pde.Heat1DExplicitSolver(equation1)

    # testing for monte carlo pricing

    ticker = 'AAPL'

    # STOCK
    historical_data = pde.HistoricalStockData(ticker)
    historical_data.fetch_stock_data( "2024-03-21","2025-03-21")
    sigma, r = historical_data.estimate_metrics()
    current_price = historical_data.get_latest_stock_price()

    equation2 = pde.BlackScholesEquation(pde.OptionType.EUROPEAN_CALL, current_price, 1, sigma, r, 100, 100, 20000)

    solver1 = pde.BlackScholesCNSolver(equation2)
    solver2 = pde.BlackScholesExplicitSolver(equation2)
    sol1 = solver1.solve()
    # sol1.plot()

    # COMPARISON
    #  look to see the corresponding option price for the expiration date and strike price
    pricing_1 = pde.BlackScholesFormula(pde.OptionType.EUROPEAN_CALL, current_price, 100, r, sigma, 1)
    pricing_2 = pde.MonteCarloPricing(pde.OptionType.EUROPEAN_CALL, current_price, 100, r, sigma, 1, 365, 10000)

    bs_price = pricing_1.get_black_scholes_merton_price()
    monte_carlo_price = pricing_2.get_monte_carlo_option_price()

    pde_price = sol1.get_result()[-1, 0]
    print(f"PDE Price: {pde_price}")
    print(f"Black-Scholes Price: {bs_price}")
    print(f"Monte-Carlo Price: {monte_carlo_price}")
    pricing_2.plot()


if __name__ == "__main__":
    main()