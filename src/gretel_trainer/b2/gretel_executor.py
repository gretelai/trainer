import time
import logging
from multiprocessing.managers import DictProxy
from typing import Callable, Optional, Tuple, Union

import pandas as pd
from gretel_client.projects.jobs import END_STATES, Job, Status
from gretel_client.projects.models import Model, read_model_config
from gretel_client.projects.projects import Project
from gretel_client.projects.records import RecordHandler

from gretel_trainer import Trainer
from gretel_trainer.b2.core import BenchmarkException, Dataset, RunIdentifier, Timer
from gretel_trainer.b2.gretel_models import GretelModel, GretelModelConfig
from gretel_trainer.b2.gretel_strategy_sdk import GretelSDKStrategy
from gretel_trainer.b2.gretel_strategy_trainer import GretelTrainerStrategy
from gretel_trainer.b2.status import NotStarted, InProgress, Completed, Failed, RunStatus

logger = logging.getLogger(__name__)


def _set_strategy(
    benchmark_model: GretelModel,
    dataset: Dataset,
    run_identifier: RunIdentifier,
    trainer: bool,
    refresh_interval: int,
    project: Optional[Project],
    project_prefix: str,
) -> Union[GretelSDKStrategy, GretelTrainerStrategy]:
    if trainer:
        return GretelTrainerStrategy(
            benchmark_model=benchmark_model,
            dataset=dataset,
            run_identifier=run_identifier,
            project_prefix=project_prefix,
        )
    else:
        return GretelSDKStrategy(
            benchmark_model=benchmark_model,
            dataset=dataset,
            run_identifier=run_identifier,
            project=project,
            refresh_interval=refresh_interval,
        )

class GretelExecutor:
    def __init__(
        self,
        benchmark_model: GretelModel,
        dataset: Dataset,
        run_identifier: RunIdentifier,
        statuses: DictProxy,
        trainer: bool,
        refresh_interval: int,
        project: Optional[Project],
        project_prefix: str,
    ):
        self.run_identifier = run_identifier
        self.statuses = statuses
        self.set_status(NotStarted())

        self._strategy = _set_strategy(
            benchmark_model=benchmark_model,
            dataset=dataset,
            trainer=trainer,
            refresh_interval=refresh_interval,
            project=project,
            project_prefix=project_prefix,
            run_identifier=run_identifier,
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
        # TODO: potentially skip (datatype, col count, etc.)
        logger.info(f"Starting model training for run `{self.run_identifier}`")
        self.set_status(InProgress(stage="train"))
        try:
            self._strategy.train()
        except:
            self.set_status(Failed(during="train", train_secs=self._strategy.train_time))

    def generate(self) -> None:
        if isinstance(self.status, Failed):
            return None

        logger.info(f"Starting synthetic data generation for run `{self.run_identifier}`")
        self.set_status(InProgress(stage="generate", train_secs=self._strategy.train_time))
        try:
            self._strategy.generate()
            self.set_status(Completed(
                sqs=self.get_sqs_score(),
                train_secs=self._strategy.train_time,
                generate_secs=self._strategy.generate_time,
                synthetic_data=self._strategy.get_synthetic_data(),
            ))
            logger.info(f"Run `{self.run_identifier}` completed successfully")
        except:
            self.set_status(Failed(
                during="generate",
                train_secs=self._strategy.train_time,
                generate_secs=self._strategy.generate_time,
            ))
            logger.info(f"Run `{self.run_identifier}` failed")
