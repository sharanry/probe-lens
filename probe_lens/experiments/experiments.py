from abc import ABC, abstractmethod

"""
Probe experiments are used to generate data for probing tasks.
"""


class ProbeExperiment(ABC):
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.data = None

    def __repr__(self) -> str:
        return self.task_name

    @abstractmethod
    def get_data(self) -> list[tuple[str, int]]:
        pass
