from pdesolvers.heat_equation_1d.heat_equation import HeatEquation
from pdesolvers.heat_equation_1d.solver.solver_strategy import SolverStrategy


class Solver:

    def __init__(self, solver : SolverStrategy) -> None:
        self._solver = solver

    def set_solver(self, solver : SolverStrategy) -> None:
        self._solver = solver

    def solve(self, problem):
        if problem is None:
            raise RuntimeError("Equation has not been instantiated")

        self._solver.solve(problem)
        return self

    def plot(self):
        self._solver.plot()