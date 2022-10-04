import time

from dataclasses import dataclass, field
from enum import Enum
from multiprocessing.managers import DictProxy
from typing import (
    Callable,
    ContextManager,
    Generic,
    List,
    Optional,
    TypeVar,
    Union,
)
from typing_extensions import Protocol

import pandas as pd


class Datatype(str, Enum):
    TABULAR_MIXED = "tabular_mixed"
    TABULAR_NUMERIC = "tabular_numeric"
    TIME_SERIES = "time_series"
    NATURAL_LANGUAGE = "natural_language"


class DataSource(Protocol):
    @property
    def name(self) -> str:
        ...

    @property
    def delimiter(self) -> str:
        ...

    @property
    def row_count(self) -> int:
        ...

    @property
    def column_count(self) -> int:
        ...

    @property
    def datatype(self) -> Datatype:
        ...

    def unwrap(self) -> ContextManager[str]:
        ...


D = TypeVar("D", bound="DataSource")


class Dataset(Protocol[D]):
    def sources(self) -> List[D]:
        ...


class Model(Protocol):
    def train(self, source: str, **kwargs) -> None:
        ...

    def generate(self, **kwargs) -> pd.DataFrame:
        ...


class Executor(Model, Protocol):
    @property
    def model_name(self) -> str:
        ...

    def runnable(self, source: DataSource) -> bool:
        ...

    def get_sqs_score(self, synthetic: pd.DataFrame, reference: str) -> int:
        ...


class Evaluator(Protocol):
    def __call__(self, synthetic: pd.DataFrame, reference: str) -> int:
        ...


ModelFactory = Callable[[], Model]


@dataclass
class Completed:
    sqs: int
    train_secs: float
    generate_secs: float
    synthetic_data: pd.DataFrame = field(repr=False)

    @property
    def display(self) -> str:
        return "Completed"


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
class Failed:
    during: str
    error: Exception
    train_secs: Optional[float] = None
    generate_secs: Optional[float] = None
    synthetic_data: Optional[pd.DataFrame] = field(default=None, repr=False)

    @property
    def display(self) -> str:
        return f"Failed ({self.during})"


RunStatus = Union[NotStarted, Skipped, InProgress, Completed, Failed]


T = TypeVar("T", bound="Executor", covariant=True)


@dataclass
class Run(Generic[T]):
    identifier: str
    source: DataSource
    executor: T


class BenchmarkException(Exception):
    pass


class Timer:
    def __init__(self):
        self.total_time = 0

    def __enter__(self):
        self.t = time.time()

    def __exit__(self, type, value, traceback):
        self.total_time = time.time() - self.t

    def duration(self) -> float:
        return round(self.total_time, 2)


TRAIN = "train"
GENERATE = "generate"
EVALUATE = "evaluate"


def execute(run: Run[Executor], results_dict: DictProxy) -> None:
    def _set_status(status: RunStatus) -> None:
        results_dict[run.identifier] = status

    if not run.executor.runnable(run.source):
        _set_status(Skipped())
        return None

    with run.source.unwrap() as path:
        _set_status(InProgress(stage=TRAIN))

        train_time = Timer()
        try:
            with train_time:
                run.executor.train(path, delimiter=run.source.delimiter)
        except Exception as e:
            _set_status(Failed(
                during=TRAIN,
                error=e,
                train_secs=train_time.duration()
            ))
            return None

        _set_status(InProgress(stage=GENERATE, train_secs=train_time.duration()))

        generate_time = Timer()
        try:
            with generate_time:
                synthetic_dataframe = run.executor.generate(
                    training_row_count=run.source.row_count
                )
        except Exception as e:
            _set_status(Failed(
                during=GENERATE,
                error=e,
                train_secs=train_time.duration(),
                generate_secs=generate_time.duration()
            ))
            return None

        _set_status(InProgress(stage=EVALUATE, train_secs=train_time.duration(), generate_secs=generate_time.duration()))

        try:
            sqs_score = run.executor.get_sqs_score(synthetic_dataframe, path)
        except Exception as e:
            _set_status(Failed(
                during=EVALUATE,
                error=e,
                train_secs=train_time.duration(),
                generate_secs=generate_time.duration(),
                synthetic_data=synthetic_dataframe
            ))
            return None

    _set_status(Completed(
        sqs=sqs_score,
        train_secs=train_time.duration(),
        generate_secs=generate_time.duration(),
        synthetic_data=synthetic_dataframe
    ))
