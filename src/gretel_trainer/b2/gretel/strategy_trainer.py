from multiprocessing.managers import DictProxy
from pathlib import Path
from typing import Optional

import pandas as pd

from gretel_trainer import Trainer
from gretel_trainer.b2.core import (
    BenchmarkException,
    Dataset,
    RunIdentifier,
    Timer,
    run_out_path,
)
from gretel_trainer.b2.gretel.models import GretelModel
from gretel_trainer.b2.status import (
    Completed,
    Failed,
    InProgress,
    NotStarted,
    RunStatus,
)


class GretelTrainerStrategy:
    def __init__(
        self,
        benchmark_model: GretelModel,
        dataset: Dataset,
        run_identifier: RunIdentifier,
        project_prefix: str,
        working_dir: Path,
    ):
        self.benchmark_model = benchmark_model
        self.dataset = dataset
        self.run_identifier = run_identifier
        self.project_prefix = project_prefix
        self.working_dir = working_dir

        self.trainer: Optional[Trainer] = None
        self.train_timer: Optional[Timer] = None
        self.generate_timer: Optional[Timer] = None

    def runnable(self, dataset: Dataset) -> bool:
        return self.benchmark_model.runnable(dataset)

    def get_train_time(self) -> Optional[float]:
        return _get_duration(self.train_timer)

    def get_generate_time(self) -> Optional[float]:
        return _get_duration(self.generate_timer)

    def train(self) -> None:
        self.trainer = Trainer(
            project_name=f"{self.project_prefix}-{self.run_identifier}",
            model_type=self.benchmark_model.trainer_model_type,
            cache_file=self.working_dir / f"{self.run_identifier}.json",
        )
        self.train_timer = Timer()
        with self.train_timer:
            self.trainer.train(self.dataset.data_source)

    def generate(self) -> None:
        if self.trainer is None:
            raise BenchmarkException("Cannot generate before training")

        self.generate_timer = Timer()
        with self.generate_timer:
            synthetic_data = self.trainer.generate(num_records=self.dataset.row_count)
        out_path = run_out_path(self.working_dir, self.run_identifier)
        synthetic_data.to_csv(out_path, index=False)

    def get_sqs_score(self) -> int:
        if self.trainer is None:
            raise BenchmarkException("Cannot get SQS score before training")

        return self.trainer.get_sqs_score()


def _get_duration(timer: Optional[Timer]) -> Optional[float]:
    if timer is None:
        return None
    else:
        return timer.duration()
