import time
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Type, Union
from typing_extensions import Protocol

import pandas as pd
import gretel_client.projects.common as client_common


class BenchmarkException(Exception):
    pass


RunIdentifier = Tuple[str, str]
DataSourceTypes = client_common.DataSourceTypes


class Datatype(str, Enum):
    tabular = "tabular"
    time_series = "time_series"
    natural_language = "natural_language"


class Dataset(Protocol):
    @property
    def data_source(self) -> DataSourceTypes:
        ...

    @property
    def datatype(self) -> Datatype:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def row_count(self) -> int:
        ...

    @property
    def column_count(self) -> int:
        ...


@dataclass
class BenchmarkConfig:
    project_display_name: str
    refresh_interval: int
    trainer: bool


class Timer:
    def __init__(self):
        self.total_time = 0

    def __enter__(self):
        self.t = time.time()

    def __exit__(self, type, value, traceback):
        self.total_time = time.time() - self.t

    def duration(self) -> float:
        return round(self.total_time, 2)
