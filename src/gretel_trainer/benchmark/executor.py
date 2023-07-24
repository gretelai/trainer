import logging
from enum import Enum
from typing import Optional, Protocol

from gretel_client.projects.models import Model
from gretel_client.projects.projects import Project

from gretel_trainer.benchmark.core import BenchmarkConfig, Dataset, log, run_out_path
from gretel_trainer.benchmark.sdk_extras import create_evaluate_model, run_evaluate


class Status(str, Enum):
    NotStarted = "Not started"
    Skipped = "Skipped"
    Training = "Training"
    Generating = "Generating"
    Evaluating = "Evaluating"
    Complete = "Complete"
    FailedTrain = "Failed (train)"
    FailedGenerate = "Failed (generate)"
    FailedEvaluate = "Failed (evaluate)"

    @property
    def can_proceed(self) -> bool:
        return not self.cannot_proceed

    @property
    def cannot_proceed(self) -> bool:
        return self in [
            Status.Skipped,
            Status.FailedTrain,
            Status.FailedGenerate,
            Status.FailedEvaluate,
        ]


class Strategy(Protocol):
    @property
    def dataset(self) -> Dataset:
        ...

    @property
    def evaluate_ref_data(self) -> str:
        ...

    def runnable(self) -> bool:
        ...

    def train(self) -> None:
        ...

    def generate(self) -> None:
        ...

    def get_train_time(self) -> Optional[float]:
        ...

    def get_generate_time(self) -> Optional[float]:
        ...


class Executor:
    def __init__(
        self,
        strategy: Strategy,
        run_identifier: str,
        evaluate_project: Project,
        config: BenchmarkConfig,
    ):
        self.strategy = strategy
        self.run_identifier = run_identifier
        self.evaluate_project = evaluate_project
        self.config = config

        self.status = Status.NotStarted
        self.exception: Optional[Exception] = None
        self.evaluate_model: Optional[Model] = None
        self.evaluate_report_json: Optional[dict] = None

    def run(self) -> None:
        self._maybe_skip()
        if self.status.can_proceed:
            self._train()
        if self.status.can_proceed:
            self._generate()
        if self.status.can_proceed:
            self._evaluate()

    def get_report_score(self, key: str) -> Optional[int]:
        if self.evaluate_report_json is None:
            return None
        return self.evaluate_report_json[key]["score"]

    def _maybe_skip(self) -> None:
        if not self.strategy.runnable():
            self._log("skipping")
            self.status = Status.Skipped

    def _train(self) -> None:
        self._log("starting model training")
        self.status = Status.Training
        try:
            self.strategy.train()
            self._log("training completed successfully")
        except Exception as e:
            self._log("training failed", level=logging.ERROR, exc_info=e)
            self.status = Status.FailedTrain
            self.exception = e

    def _generate(self) -> None:
        self._log("starting synthetic data generation")
        self.status = Status.Generating
        try:
            self.strategy.generate()
            self._log("synthetic data generation completed successfully")
        except Exception as e:
            self._log(
                "synthetic data generation failed", level=logging.ERROR, exc_info=e
            )
            self.status = Status.FailedGenerate
            self.exception = e

    def _evaluate(self) -> None:
        self._log("starting evaluation")
        self.status = Status.Evaluating

        records_file = str(run_out_path(self.config.working_dir, self.run_identifier))
        records_artifact_key = self.evaluate_project.upload_artifact(records_file)
        try:
            self.evaluate_model = create_evaluate_model(
                project=self.evaluate_project,
                data_source=records_artifact_key,
                ref_data=self.strategy.evaluate_ref_data,
                run_identifier=self.run_identifier,
            )
            self.evaluate_report_json = run_evaluate(
                evaluate_model=self.evaluate_model,
                run_identifier=self.run_identifier,
                wait=self.config.refresh_interval,
            )
            self._log("evaluation completed successfully")
            self.status = Status.Complete
        except Exception as e:
            self._log("evaluation failed", level=logging.ERROR, exc_info=e)
            self.status = Status.FailedEvaluate
            self.exception = e
        finally:
            try:
                self.evaluate_project.delete_artifact(records_artifact_key)
            except:
                pass

    def _log(
        self, msg: str, level: int = logging.INFO, exc_info: Optional[Exception] = None
    ) -> None:
        log(self.run_identifier, msg, level, exc_info)
