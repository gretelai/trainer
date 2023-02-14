from multiprocessing.managers import DictProxy
from pathlib import Path
from typing import Optional

import pandas as pd

from gretel_trainer import Trainer
from gretel_trainer.b2.core import BenchmarkException, Dataset, RunIdentifier, Timer
from gretel_trainer.b2.gretel_models import GretelModel
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
        self.synthetic_data: Optional[pd.DataFrame] = None
        self.train_time: Optional[float] = None
        self.generate_time: Optional[float] = None

        self.working_dir.mkdir(exist_ok=True)

    def train(self) -> None:
        self.trainer = Trainer(
            project_name=f"{self.project_prefix}-{self.run_identifier}",
            model_type=self.benchmark_model.trainer_model_type,
            cache_file=self.working_dir / f"{self.run_identifier}.json",
        )
        train_time = Timer()
        try:
            with train_time:
                data_source = self.dataset.data_source
                if isinstance(data_source, pd.DataFrame):
                    csv_path = self.working_dir / f"{self.run_identifier}.csv"
                    data_source.to_csv(csv_path, index=False)
                    data_source = csv_path
                self.trainer.train(data_source)
            self.train_time = train_time.duration()
        except Exception as e:
            raise BenchmarkException("Training failed") from e

    def generate(self) -> None:
        if self.trainer is None or self.train_time is None:
            raise BenchmarkException("Cannot generate before training")

        generate_time = Timer()
        try:
            with generate_time:
                self.synthetic_data = self.trainer.generate(
                    num_records=self.dataset.row_count
                )
            self.generate_time = generate_time.duration()
            return None
        except Exception as e:
            raise BenchmarkException("Generate failed") from e

    def get_sqs_score(self) -> int:
        if self.trainer is None:
            raise BenchmarkException("Cannot get SQS score before training")

        return self.trainer.get_sqs_score()

    def get_synthetic_data(self) -> pd.DataFrame:
        if self.synthetic_data is None:
            raise BenchmarkException("Cannot get synthetic data before generating")

        return self.synthetic_data
