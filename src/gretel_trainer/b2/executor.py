from enum import Enum
from multiprocessing.managers import DictProxy
from pathlib import Path
from typing import Optional, Protocol

from gretel_trainer.b2.core import BenchmarkConfig, Dataset, log, run_out_path


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

    def runnable(self) -> bool:
        ...

    def train(self) -> None:
        ...

    def generate(self) -> None:
        ...

    def evaluate(self) -> None:
        ...

    def get_sqs_score(self) -> Optional[int]:
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
    ):
        self.strategy = strategy
        self.run_identifier = run_identifier
        self.status = Status.NotStarted
        self.exception: Optional[Exception] = None

    def run(self) -> None:
        self._maybe_skip()
        if self.status.can_proceed:
            self._train()
        if self.status.can_proceed:
            self._generate()
        if self.status.can_proceed:
            self._evaluate()

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
            self._log("training failed")
            self.status = Status.FailedTrain
            self.exception = e

    def _generate(self) -> None:
        self._log("starting synthetic data generation")
        self.status = Status.Generating
        try:
            self.strategy.generate()
            self._log("synthetic data generation completed successfully")
        except Exception as e:
            self._log("synthetic data generation failed")
            self.status = Status.FailedGenerate
            self.exception = e

    def _evaluate(self) -> None:
        self._log("starting evaluation")
        self.status = Status.Evaluating

        try:
            self.strategy.evaluate()
            sqs = self.strategy.get_sqs_score()
            self._log("evaluation completed successfully")
            self.status = Status.Complete
        except Exception as e:
            self._log("evaluation failed")
            self.status = Status.FailedEvaluate
            self.exception = e

    def _log(self, msg: str) -> None:
        log(self.run_identifier, msg)
