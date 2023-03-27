from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from gretel_client.projects.projects import Project

from gretel_trainer import Trainer
from gretel_trainer.b2.core import (
    BenchmarkConfig,
    BenchmarkException,
    Dataset,
    Timer,
    run_out_path,
)
from gretel_trainer.b2.gretel.models import GretelModel


class GretelTrainerStrategy:
    def __init__(
        self,
        benchmark_model: GretelModel,
        dataset: Dataset,
        run_identifier: str,
        config: BenchmarkConfig,
    ):
        self.benchmark_model = benchmark_model
        self.dataset = dataset
        self.run_identifier = run_identifier
        self.config = config

        self.trainer: Optional[Trainer] = None
        self.train_timer: Optional[Timer] = None
        self.generate_timer: Optional[Timer] = None

    def runnable(self) -> bool:
        return self.benchmark_model.runnable(self.dataset)

    def get_train_time(self) -> Optional[float]:
        return _get_duration(self.train_timer)

    def get_generate_time(self) -> Optional[float]:
        return _get_duration(self.generate_timer)

    def train(self) -> None:
        self.trainer = Trainer(
            project_name=self._project_name,
            model_type=self.benchmark_model.trainer_model_type,
            cache_file=self.config.working_dir / f"{self.run_identifier}.json",
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
        synthetic_data.to_csv(self._synthetic_data_path, index=False)

    @property
    def evaluate_ref_data(self) -> str:
        return self.dataset.data_source

    @property
    def _synthetic_data_path(self) -> Path:
        return run_out_path(self.config.working_dir, self.run_identifier)

    @property
    def _project_name(self) -> str:
        prefix = self.config.trainer_project_prefix
        run_id = self.run_identifier
        name = f"{prefix}-{run_id}"
        return name.replace("_", "-")


def _get_duration(timer: Optional[Timer]) -> Optional[float]:
    if timer is None:
        return None
    else:
        return timer.duration()
