import logging
from multiprocessing.managers import DictProxy
from pathlib import Path
from typing import Optional, Protocol

from gretel_trainer.b2.core import (
    BenchmarkConfig,
    Dataset,
    RunIdentifier,
    run_out_path,
)
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

    def get_sqs_score(self) -> int:
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
            logger.info(f"Skipping model training for run `{self.run_identifier}`")
            self.set_status(Skipped())
            return None

        logger.info(f"Starting model training for run `{self.run_identifier}`")
        self.set_status(InProgress(stage="train"))
        try:
            self.strategy.train()
            logger.info(
                f"Training completed successfully for run `{self.run_identifier}`"
            )
        except Exception as e:
            logger.info(f"Training failed for run `{self.run_identifier}`")
            self.set_status(
                Failed(
                    during="train",
                    error=e,
                    train_secs=self.strategy.get_train_time(),
                )
            )

    def generate(self) -> None:
        if isinstance(self.status, (Skipped, Failed)):
            return None

        logger.info(
            f"Starting synthetic data generation for run `{self.run_identifier}`"
        )
        self.set_status(
            InProgress(
                stage="generate",
                train_secs=self.strategy.get_train_time(),
            )
        )
        try:
            self.strategy.generate()
            logger.info(
                f"Synthetic data generation completed successfully for run `{self.run_identifier}`"
            )
        except Exception as e:
            logger.info(
                f"Synthetic data generation failed for run `{self.run_identifier}`"
            )
            self.set_status(
                Failed(
                    during="generate",
                    error=e,
                    train_secs=self.strategy.get_train_time(),
                    generate_secs=self.strategy.get_generate_time(),
                )
            )

    def evaluate(self) -> None:
        if isinstance(self.status, (Skipped, Failed)):
            return None

        logger.info(
            f"Starting evaluation for run `{self.run_identifier}`"
        )
        self.set_status(
            InProgress(
                stage="evaluate",
                train_secs=self.strategy.get_train_time(),
                generate_secs=self.strategy.get_generate_time(),
                synthetic_data=self._synthetic_data_path,
            )
        )

        try:
            sqs = self.strategy.get_sqs_score()
            self.set_status(
                Completed(
                    sqs=sqs,
                    train_secs=self.strategy.get_train_time(),
                    generate_secs=self.strategy.get_generate_time(),
                    synthetic_data=self._synthetic_data_path,
                )
            )
        except Exception as e:
            logger.info(
                f"Evaluation failed for run `{self.run_identifier}`"
            )
            self.set_status(
                Failed(
                    during="evaluate",
                    error=e,
                    train_secs=self.strategy.get_train_time(),
                    generate_secs=self.strategy.get_generate_time(),
                    synthetic_data=self._synthetic_data_path,
                )
            )

    @property
    def _synthetic_data_path(self) -> Path:
        return run_out_path(self.config.working_dir, self.run_identifier)
