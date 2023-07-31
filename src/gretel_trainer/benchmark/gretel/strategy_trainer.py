from pathlib import Path
from typing import Optional

from gretel_trainer import Trainer
from gretel_trainer.benchmark.core import (
    BenchmarkConfig,
    BenchmarkException,
    Dataset,
    Timer,
    run_out_path,
)
from gretel_trainer.benchmark.gretel.models import GretelModel
from gretel_trainer.benchmark.job_spec import JobSpec


class GretelTrainerStrategy:
    def __init__(
        self,
        job_spec: JobSpec[GretelModel],
        run_identifier: str,
        project_name: str,
        config: BenchmarkConfig,
    ):
        self.job_spec = job_spec
        self.run_identifier = run_identifier
        self.project_name = project_name
        self.config = config

        self.trainer: Optional[Trainer] = None
        self.train_timer: Optional[Timer] = None
        self.generate_timer: Optional[Timer] = None

    @property
    def dataset(self) -> Dataset:
        return self.job_spec.dataset

    @property
    def benchmark_model(self) -> GretelModel:
        return self.job_spec.model

    def runnable(self) -> bool:
        return self.benchmark_model.runnable(self.dataset)

    def get_train_time(self) -> Optional[float]:
        return _get_duration(self.train_timer)

    def get_generate_time(self) -> Optional[float]:
        return _get_duration(self.generate_timer)

    def train(self) -> None:
        self.trainer = Trainer(
            project_name=self.project_name,
            model_type=self.benchmark_model.trainer_model_type,
            cache_file=str(self.config.working_dir / f"{self.run_identifier}.json"),
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


def _get_duration(timer: Optional[Timer]) -> Optional[float]:
    if timer is None:
        return None
    else:
        return timer.duration()
