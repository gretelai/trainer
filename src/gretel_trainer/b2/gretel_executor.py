import logging
import time
from multiprocessing.managers import DictProxy
from typing import Callable, Optional, Tuple, Union

import pandas as pd
from gretel_client.projects.jobs import END_STATES, Job, Status
from gretel_client.projects.models import Model, read_model_config
from gretel_client.projects.projects import Project
from gretel_client.projects.records import RecordHandler

from gretel_trainer import Trainer
from gretel_trainer.b2.core import (
    BenchmarkConfig,
    BenchmarkException,
    Dataset,
    RunIdentifier,
    Timer,
)
from gretel_trainer.b2.gretel_models import GretelModel, GretelModelConfig
from gretel_trainer.b2.gretel_strategy_sdk import GretelSDKStrategy
from gretel_trainer.b2.gretel_strategy_trainer import GretelTrainerStrategy
from gretel_trainer.b2.status import (
    Completed,
    Failed,
    InProgress,
    NotStarted,
    RunStatus,
    Skipped,
)

logger = logging.getLogger(__name__)


def _set_strategy(
    benchmark_model: GretelModel,
    dataset: Dataset,
    run_identifier: RunIdentifier,
    config: BenchmarkConfig,
    project: Optional[Project],
) -> Union[GretelSDKStrategy, GretelTrainerStrategy]:
    if config.trainer:
        return GretelTrainerStrategy(
            benchmark_model=benchmark_model,
            dataset=dataset,
            run_identifier=run_identifier,
            project_prefix=config.project_display_name,
            working_dir=config.working_dir,
        )
    else:
        return GretelSDKStrategy(
            benchmark_model=benchmark_model,
            dataset=dataset,
            run_identifier=run_identifier,
            project=project,
            refresh_interval=config.refresh_interval,
        )


class GretelExecutor:
    def __init__(
        self,
        benchmark_model: GretelModel,
        dataset: Dataset,
        run_identifier: RunIdentifier,
        statuses: DictProxy,
        config: BenchmarkConfig,
        project: Optional[Project],
    ):
        self.run_identifier = run_identifier
        self.statuses = statuses
        self.set_status(NotStarted())

        self._strategy = _set_strategy(
            benchmark_model=benchmark_model,
            dataset=dataset,
            run_identifier=run_identifier,
            config=config,
            project=project,
        )

    @property
    def benchmark_model(self) -> GretelModel:
        return self._strategy.benchmark_model

    @property
    def dataset(self) -> Dataset:
        return self._strategy.dataset

    def get_sqs_score(self) -> int:
        return self._strategy.get_sqs_score()

    def get_synthetic_data(self) -> pd.DataFrame:
        return self._strategy.get_synthetic_data()

    def set_status(self, status: RunStatus) -> None:
        self.status = status
        self.statuses[self.run_identifier] = status

    def train(self) -> None:
        if not self.benchmark_model.runnable(self.dataset):
            logger.info(f"Skipping model training for run `{self.run_identifier}`")
            self.set_status(Skipped())
            return None

        logger.info(f"Starting model training for run `{self.run_identifier}`")
        self.set_status(InProgress(stage="train"))
        try:
            self._strategy.train()
        except:
            self.set_status(
                Failed(during="train", train_secs=self._strategy.train_time)
            )

    def generate(self) -> None:
        if isinstance(self.status, (Skipped, Failed)):
            return None

        logger.info(
            f"Starting synthetic data generation for run `{self.run_identifier}`"
        )
        self.set_status(
            InProgress(stage="generate", train_secs=self._strategy.train_time)
        )
        try:
            self._strategy.generate()
            self.set_status(
                Completed(
                    sqs=self.get_sqs_score(),
                    train_secs=self._strategy.train_time,
                    generate_secs=self._strategy.generate_time,
                    synthetic_data=self._strategy.get_synthetic_data(),
                )
            )
            logger.info(f"Run `{self.run_identifier}` completed successfully")
        except:
            self.set_status(
                Failed(
                    during="generate",
                    train_secs=self._strategy.train_time,
                    generate_secs=self._strategy.generate_time,
                )
            )
            logger.info(f"Run `{self.run_identifier}` failed")
