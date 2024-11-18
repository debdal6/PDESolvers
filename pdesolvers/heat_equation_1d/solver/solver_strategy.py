from abc import ABC, abstractmethod
import numpy as np
from matplotlib import pyplot as plt

from pdesolvers.heat_equation_1d.heat_equation import HeatEquation


class SolverStrategy(ABC):

    @abstractmethod
    def solve(self, equation : HeatEquation):
        pass

    def plot(self):
        pass