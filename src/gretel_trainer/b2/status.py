from dataclasses import dataclass, field
from typing import Optional, Union

import pandas as pd


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
class InProgress:
    stage: str
    train_secs: Optional[float] = None
    generate_secs: Optional[float] = None

    @property
    def display(self) -> str:
        return f"In progress ({self.stage})"


@dataclass
class Completed:
    sqs: int
    train_secs: Optional[float]
    generate_secs: Optional[float]
    synthetic_data: pd.DataFrame = field(repr=False)

    @property
    def display(self) -> str:
        return "Completed"


@dataclass
class Failed:
    during: str
    error: Optional[Exception] = None
    train_secs: Optional[float] = None
    generate_secs: Optional[float] = None
    synthetic_data: Optional[pd.DataFrame] = field(default=None, repr=False)

    @property
    def display(self) -> str:
        return f"Failed ({self.during})"


RunStatus = Union[NotStarted, Skipped, InProgress, Completed, Failed]
