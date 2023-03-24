import logging
from multiprocessing.managers import DictProxy
from pathlib import Path
from typing import Optional, Protocol

from gretel_trainer.b2.core import BenchmarkConfig, Dataset, RunIdentifier, run_out_path
from gretel_trainer.b2.status import (
    Completed,
    Failed,
    InProgress,
    NotStarted,
    RunStatus,
    Skipped,
)

logger = logging.getLogger(__name__)


class Strategy(Protocol):
    def runnable(self, dataset: Dataset) -> bool:
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
        model_name: str,
        dataset: Dataset,
        run_identifier: RunIdentifier,
        statuses: DictProxy,
        config: BenchmarkConfig,
    ):
        self.strategy = strategy
        self.model_name = model_name
        self.dataset = dataset
        self.run_identifier = run_identifier
        self.statuses = statuses
        self.config = config
        self.set_status(NotStarted())

    def set_status(self, status: RunStatus) -> None:
        self.status = status
        self.statuses[self.run_identifier] = status

    def train(self) -> None:
        if not self.strategy.runnable(self.dataset):
            logger.info(f"Skipping model training for run `{self.run_identifier}`")
            self.set_status(Skipped())
            return None

        logger.info(f"Starting model training for run `{self.run_identifier}`")
        self._in_progress("training")
        try:
            self.strategy.train()
            logger.info(
                f"Training completed successfully for run `{self.run_identifier}`"
            )
            self._in_progress("trained")
        except Exception as e:
            logger.info(f"Training failed for run `{self.run_identifier}`")
            self._failed("train", e)

    def generate(self) -> None:
        if isinstance(self.status, (Skipped, Failed)):
            return None

        logger.info(
            f"Starting synthetic data generation for run `{self.run_identifier}`"
        )
        self._in_progress("generating")
        try:
            self.strategy.generate()
            logger.info(
                f"Synthetic data generation completed successfully for run `{self.run_identifier}`"
            )
            self._in_progress("generated")
        except Exception as e:
            logger.info(
                f"Synthetic data generation failed for run `{self.run_identifier}`"
            )
            self._failed("generate", e)

    def evaluate(self) -> None:
        if isinstance(self.status, (Skipped, Failed)):
            return None

        logger.info(f"Starting evaluation for run `{self.run_identifier}`")
        self._in_progress("evaluating")

        try:
            self.strategy.evaluate()
            sqs = self.strategy.get_sqs_score()
            logger.info(
                f"Evaluation completed successfully for run `{self.run_identifier}`"
            )
            self.set_status(
                Completed(
                    sqs=sqs,
                    train_secs=self.strategy.get_train_time(),
                    generate_secs=self.strategy.get_generate_time(),
                )
            )
        except Exception as e:
            logger.info(f"Evaluation failed for run `{self.run_identifier}`")
            self._failed("evaluate", e)

    def _in_progress(self, stage: str) -> None:
        self.set_status(
            InProgress(
                stage=stage,
                train_secs=self.strategy.get_train_time(),
                generate_secs=self.strategy.get_generate_time(),
            )
        )

    def _failed(self, during: str, error: Exception) -> None:
        self.set_status(
            Failed(
                during=during,
                error=error,
                train_secs=self.strategy.get_train_time(),
                generate_secs=self.strategy.get_generate_time(),
            )
        )
