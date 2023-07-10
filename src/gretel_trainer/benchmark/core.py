import csv
import logging
import time
from dataclasses import InitVar, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Union

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


def _default_name() -> str:
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"benchmark-{current_time}"


@dataclass
class BenchmarkConfig:
    project_display_name: str = field(default_factory=_default_name)
    refresh_interval: int = 15
    trainer: bool = False
    work_dir: InitVar[Optional[Union[str, Path]]] = None
    working_dir: Path = field(init=False)
    additional_report_scores: list[str] = field(default_factory=list)

    def __post_init__(self, work_dir):
        if work_dir is None:
            work_dir = self.project_display_name
        self.working_dir = Path(work_dir)


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
