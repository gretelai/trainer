from pathlib import Path
from typing import Any, Dict, Optional

from gretel_client.projects.projects import Project

from gretel_trainer.b2.core import (
    BenchmarkConfig,
    Dataset,
    RunIdentifier,
    Timer,
    run_out_path,
)
from gretel_trainer.b2.custom.models import CustomModel
from gretel_trainer.b2.gretel.sdk_extras import run_evaluate


class CustomStrategy:
    def __init__(
        self,
        benchmark_model: CustomModel,
        dataset: Dataset,
        run_identifier: RunIdentifier,
        evaluate_project: Project,
        config: BenchmarkConfig,
    ):
        self.benchmark_model = benchmark_model
        self.dataset = dataset
        self.run_identifier = run_identifier
        self.evaluate_project = evaluate_project
        self.config = config

        self.train_timer: Optional[Timer] = None
        self.generate_timer: Optional[Timer] = None
        self.evaluate_report_json: Optional[Dict[str, Any]] = None

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

    def evaluate(self) -> None:
        self.evaluate_report_json = run_evaluate(
            project=self.evaluate_project,
            data_source=str(self._synthetic_data_path),
            ref_data=self.dataset.data_source,
            run_identifier=self.run_identifier,
            wait=self.config.refresh_interval,
        )

    def get_sqs_score(self) -> Optional[int]:
        if self.evaluate_report_json is None:
            return None
        else:
            return self.evaluate_report_json["synthetic_data_quality_score"]["score"]

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
