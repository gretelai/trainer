import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Tuple, Type, Union

import pandas as pd
from typing_extensions import Protocol


class BenchmarkException(Exception):
    pass


class RunIdentifier(Tuple[str, str]):
    def __repr__(self) -> str:
        return f"{self[0]}-{self[1]}"


class Datatype(str, Enum):
    tabular = "tabular"
    time_series = "time_series"
    natural_language = "natural_language"


@dataclass
class Dataset:
    name: str
    datatype: Datatype
    data_source: str
    row_count: int
    column_count: int


@dataclass
class BenchmarkConfig:
    project_display_name: str
    refresh_interval: int
    trainer: bool
    working_dir: Path
    timestamp: str


class Timer:
    def __init__(self):
        self.total_time = 0

    def __enter__(self):
        self.t = time.time()

    def __exit__(self, type, value, traceback):
        self.total_time = time.time() - self.t

    def duration(self) -> float:
        return round(self.total_time, 2)


def run_out_path(working_dir: Path, run_identifier: RunIdentifier) -> Path:
    return working_dir / "out" / f"{run_identifier}.csv"
