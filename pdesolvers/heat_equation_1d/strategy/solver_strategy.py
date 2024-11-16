from abc import ABC, abstractmethod

class SolverStrategy(ABC):

    @abstractmethod
    def solve(self, context):
        pass
