from pathlib import Path
from typing import Optional

from gretel_trainer.benchmark.core import BenchmarkConfig, Dataset, Timer, run_out_path
from gretel_trainer.benchmark.custom.models import CustomModel


class CustomStrategy:
    def __init__(
        self,
        benchmark_model: CustomModel,
        dataset: Dataset,
        run_identifier: str,
        config: BenchmarkConfig,
        artifact_key: Optional[str] = None,
    ):
        self.benchmark_model = benchmark_model
        self.dataset = dataset
        self.run_identifier = run_identifier
        self.config = config
        self.artifact_key = artifact_key

        self.train_timer: Optional[Timer] = None
        self.generate_timer: Optional[Timer] = None

    def train(self) -> None:
        self.train_timer = Timer()
        with self.train_timer:
            self.benchmark_model.train(self.dataset)

    def generate(self) -> None:
        self.generate_timer = Timer()
        with self.generate_timer:
            synthetic_df = self.benchmark_model.generate(
                num_records=self.dataset.row_count,
            )
            synthetic_df.to_csv(self._synthetic_data_path, index=False)

    @property
    def evaluate_ref_data(self) -> str:
        if self.artifact_key is not None:
            return self.artifact_key
        return self.dataset.data_source

    @property
    def _synthetic_data_path(self) -> Path:
        return run_out_path(self.config.working_dir, self.run_identifier)

    def runnable(self) -> bool:
        return True

    def get_train_time(self) -> Optional[float]:
        return _get_duration(self.train_timer)

    def get_generate_time(self) -> Optional[float]:
        return _get_duration(self.generate_timer)


def _get_duration(timer: Optional[Timer]) -> Optional[float]:
    if timer is None:
        return None
    else:
        return timer.duration()
