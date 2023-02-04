from dataclasses import dataclass
from enum import Enum
from typing import Type, Union
from typing_extensions import Protocol

import pandas as pd
import gretel_client.projects.common as client_common


class BenchmarkException(Exception):
    pass


DataSourceTypes = client_common.DataSourceTypes


class Dataset(Protocol):
    @property
    def data_source(self) -> DataSourceTypes:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def row_count(self) -> int:
        ...


@dataclass
class BenchmarkConfig:
    project_display_name: str = "benchmark"
    refresh_interval: int = 60


class Datatype(str, Enum):
    tabular = "tabular"
    time_series = "time_series"
    natural_language = "natural_language"
