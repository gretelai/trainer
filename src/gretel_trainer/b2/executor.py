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
        dataset: Dataset,
        run_identifier: RunIdentifier,
        statuses: DictProxy,
        config: BenchmarkConfig,
    ):
        self.strategy = strategy
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
            logger.info(f"{self.run_identifier} - skipping")
            self.set_status(Skipped())
            return None

        logger.info(f"{self.run_identifier} - starting model training")
        self._in_progress("training")
        try:
            self.strategy.train()
            logger.info(f"{self.run_identifier} - training completed successfully")
            self._in_progress("trained")
        except Exception as e:
            logger.info(f"{self.run_identifier} - training failed")
            self._failed("train", e)

    def generate(self) -> None:
        if isinstance(self.status, (Skipped, Failed)):
            return None

        logger.info(f"{self.run_identifier} - starting synthetic data generation")
        self._in_progress("generating")
        try:
            self.strategy.generate()
            logger.info(
                f"{self.run_identifier} - synthetic data generation completed successfully"
            )
            self._in_progress("generated")
        except Exception as e:
            logger.info(f"{self.run_identifier} - synthetic data generation failed")
            self._failed("generate", e)

    def evaluate(self) -> None:
        if isinstance(self.status, (Skipped, Failed)):
            return None

        logger.info(f"{self.run_identifier} - starting evaluation")
        self._in_progress("evaluating")

        try:
            self.strategy.evaluate()
            sqs = self.strategy.get_sqs_score()
            logger.info(f"{self.run_identifier} - evaluation completed successfully")
            self.set_status(
                Completed(
                    sqs=sqs,
                    train_secs=self.strategy.get_train_time(),
                    generate_secs=self.strategy.get_generate_time(),
                )
            )
        except Exception as e:
            logger.info(f"{self.run_identifier} - evaluation failed")
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
