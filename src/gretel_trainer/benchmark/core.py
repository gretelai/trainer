import csv
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import smart_open

logger = logging.getLogger(__name__)


class BenchmarkException(Exception):
    pass


class Datatype(str, Enum):
    TABULAR = "tabular"
    TIME_SERIES = "time_series"
    NATURAL_LANGUAGE = "natural_language"


@dataclass
class Dataset:
    name: str
    datatype: Datatype
    data_source: str
    row_count: int
    column_count: int
    public: bool = field(default=False)


@dataclass
class BenchmarkConfig:
    project_display_name: str
    refresh_interval: int
    trainer: bool
    working_dir: Path


class Timer:
    def __init__(self):
        self.total_time = 0

    def __enter__(self):
        self.t = time.time()

    def __exit__(self, type, value, traceback):
        self.total_time = time.time() - self.t

    def duration(self) -> float:
        return round(self.total_time, 3)


def run_out_path(working_dir: Path, run_identifier: str) -> Path:
    return working_dir / f"synth_{run_identifier}.csv"


def log(run_identifier: str, msg: str) -> None:
    logger.info(f"{run_identifier} - {msg}")


def get_data_shape(path: str, delimiter: str = ",") -> tuple[int, int]:
    with smart_open.open(path) as f:
        reader = csv.reader(f, delimiter=delimiter)
        cols = len(next(reader))
        # We just read the header row to get cols,
        # so the remaining rows are all data
        rows = sum(1 for _ in f)
    return (rows, cols)
