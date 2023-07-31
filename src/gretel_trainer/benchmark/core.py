import csv
import logging
import time
from dataclasses import dataclass, field
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


class BenchmarkConfig:
    def __init__(
        self,
        project_display_name: Optional[str] = None,
        refresh_interval: int = 15,
        trainer: bool = False,
        working_dir: Optional[Union[str, Path]] = None,
        additional_report_scores: Optional[list[str]] = None,
    ):
        self.project_display_name = project_display_name or _default_name()
        self.working_dir = Path(working_dir or self.project_display_name)
        self.refresh_interval = refresh_interval
        self.trainer = trainer
        self.additional_report_scores = additional_report_scores or []


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


def log(
    run_identifier: str,
    msg: str,
    level: int = logging.INFO,
    exc_info: Optional[Exception] = None,
) -> None:
    logger.log(level, f"{run_identifier} - {msg}", exc_info=exc_info)


def get_data_shape(path: str, delimiter: str = ",") -> tuple[int, int]:
    with smart_open.open(path) as f:
        reader = csv.reader(f, delimiter=delimiter)
        cols = len(next(reader))
        # We just read the header row to get cols,
        # so the remaining rows are all data
        rows = sum(1 for _ in f)
    return (rows, cols)
