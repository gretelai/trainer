import logging
from multiprocessing.managers import DictProxy
from pathlib import Path
from typing import Optional

import pandas as pd

from gretel_trainer.b2.core import BenchmarkException, Dataset, RunIdentifier, Timer, run_out_path
from gretel_trainer.b2.custom_models import CustomModel
from gretel_trainer.b2.status import (
    Completed,
    Failed,
    InProgress,
    NotStarted,
    RunStatus,
)

logger = logging.getLogger(__name__)


class CustomExecutor:
    def __init__(
        self,
        model: CustomModel,
        dataset: Dataset,
        run_identifier: RunIdentifier,
        working_dir: Path,
        statuses: DictProxy,
    ):
        self.model = model
        self.dataset = dataset
        self.run_identifier = run_identifier
        self.working_dir = working_dir
        self.statuses = statuses
        self.set_status(NotStarted())

        self.train_time: Optional[float] = None
        self.generate_time: Optional[float] = None

    @property
    def model_name(self) -> str:
        return type(self.model).__name__

    def set_status(self, status: RunStatus) -> None:
        self.status = status
        self.statuses[self.run_identifier] = status

    def train(self) -> None:
        logger.info(f"Startig model training for run `{self.run_identifier}`")
        self.set_status(InProgress(stage="train"))
        train_time = Timer()
        with train_time:
            try:
                self.model.train(self.dataset)
                self.train_time = train_time.duration()
            except Exception as e:
                self.train_time = train_time.duration()
                self.set_status(
                    Failed(during="train", error=e, train_secs=train_time.duration())
                )

    def generate(self) -> None:
        logger.info(
            f"Starting synthetic data generation for run `{self.run_identifier}`"
        )
        self.set_status(InProgress(stage="generate", train_secs=self.train_time))
        generate_time = Timer()
        with generate_time:
            try:
                preferred_out_path = run_out_path(self.working_dir, self.run_identifier)
                synthetic_data_path = self.model.generate(
                    num_records=self.dataset.row_count,
                    preferred_out_path=preferred_out_path,
                )
                self.generate_time = generate_time.duration()
                self.set_status(
                    Completed(
                        sqs=self.get_sqs_score(),
                        train_secs=self.train_time,
                        generate_secs=self.generate_time,
                        synthetic_data=synthetic_data_path,
                    )
                )
                logger.info(f"Run `{self.run_identifier}` completed successfully")
            except Exception as e:
                self.generate_time = generate_time.duration()
                self.set_status(
                    Failed(
                        during="generate",
                        error=e,
                        train_secs=self.train_time,
                        generate_secs=self.generate_time,
                    )
                )
                logger.info(f"Run `{self.run_identifier}` failed")

    def get_sqs_score(self) -> int:
        # TODO: run a QualityReport here
        return 0
