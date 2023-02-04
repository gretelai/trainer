from dataclasses import dataclass
from typing import Union


@dataclass
class NotStarted:
    @property
    def display(self) -> str:
        return "Not started"


@dataclass
class Skipped:
    @property
    def display(self) -> str:
        return "Skipped"


@dataclass
class Completed:
    @property
    def display(self) -> str:
        return "Completed"


RunResult = Union[NotStarted, Skipped, Completed]
