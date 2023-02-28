import logging
from multiprocessing.managers import DictProxy
from pathlib import Path
from typing import Optional

import pandas as pd

from gretel_trainer.b2.core import (
    BenchmarkException,
    Dataset,
    RunIdentifier,
    Timer,
    run_out_path,
)
from gretel_trainer.b2.custom.models import CustomModel
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
        model_name: str,
        dataset: Dataset,
        run_identifier: RunIdentifier,
        working_dir: Path,
        statuses: DictProxy,
    ):
        self.model = model
        self.model_name = model_name
        self.dataset = dataset
        self.run_identifier = run_identifier
        self.working_dir = working_dir
        self.statuses = statuses
        self.set_status(NotStarted())

        self.train_time: Optional[float] = None
        self.generate_time: Optional[float] = None

    def set_status(self, status: RunStatus) -> None:
        self.status = status
        self.statuses[self.run_identifier] = status

    def train(self) -> None:
        logger.info(f"Starting model training for run `{self.run_identifier}`")
        self.set_status(InProgress(stage="train"))
        train_time = Timer()
        try:
            with train_time:
                self.model.train(self.dataset)
        except Exception as e:
            self.train_time = train_time.duration()
            logger.info(f"Training failed for run `{self.run_identifier}`")
            self.set_status(
                Failed(during="train", error=e, train_secs=train_time.duration())
            )

        self.train_time = train_time.duration()
        logger.info(f"Training completed successfully for run `{self.run_identifier}`")

    def generate(self) -> None:
        logger.info(
            f"Starting synthetic data generation for run `{self.run_identifier}`"
        )
        self.set_status(InProgress(stage="generate", train_secs=self.train_time))
        synthetic_data_path = run_out_path(self.working_dir, self.run_identifier)
        generate_time = Timer()
        try:
            with generate_time:
                synthetic_df = self.model.generate(
                    num_records=self.dataset.row_count,
                )
                synthetic_df.to_csv(synthetic_data_path, index=False)
        except Exception as e:
            self.generate_time = generate_time.duration()
            logger.info(
                f"Synthetic data generation failed for run `{self.run_identifier}`"
            )
            self.set_status(
                Failed(
                    during="generate",
                    error=e,
                    train_secs=self.train_time,
                    generate_secs=self.generate_time,
                )
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
        logger.info(
            f"Synthetic data generation completed successfully for run `{self.run_identifier}`"
        )

    def get_sqs_score(self) -> int:
        # TODO: run a QualityReport here
        return 0
