import numpy as np
from heat_equation import HeatEquation
from pdesolvers.heat_equation_1d.solver.cn_solver import HeatEquationCNSolver
from pdesolvers.heat_equation_1d.solver.explicit_solver import HeatEquationExplicitSolver


def main():

    equation = (((HeatEquation(1, 100,30,10000, 0.01)
                  .set_initial_temp(lambda x: np.sin(np.pi * x) + 5))
                 .set_left_boundary_temp(lambda t: 20 * np.sin(np.pi * t) + 5))
                .set_right_boundary_temp(lambda t: t + 5))

    explicit_solver = HeatEquationExplicitSolver()
    cn_solver = HeatEquationCNSolver()
    equation.set_solver(cn_solver)
    equation.solve().plot()



if __name__ == "__main__":
    main()