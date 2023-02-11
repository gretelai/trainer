import logging
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.managers import DictProxy
from typing import Dict, List, Optional, Tuple, Type, Union

import pandas as pd
from gretel_client import configure_session
from gretel_client.projects import create_project

from gretel_trainer.b2.core import BenchmarkConfig, Dataset, RunIdentifier
from gretel_trainer.b2.custom_models import CustomModel
from gretel_trainer.b2.gretel_models import GretelModel
from gretel_trainer.b2.gretel_sdk_executor import GretelSDKExecutor
# from gretel_trainer.b2.gretel_trainer_executor import GretelTrainerExecutor


logger = logging.getLogger(__name__)


ModelTypes = Union[Type[CustomModel], Type[GretelModel]]
Executor = GretelSDKExecutor #Union[GretelSDKExecutor, GretelTrainerExecutor, CustomExecutor]


class Comparison:
    def __init__(
        self,
        *,
        datasets: List[Dataset],
        models: List[ModelTypes],
        config: Optional[BenchmarkConfig] = None,
    ):
        self.datasets = datasets
        self.gretel_models = [m for m in models if issubclass(m, GretelModel)]
        self.custom_models = [m for m in models if not m in self.gretel_models]
        self.config = config or BenchmarkConfig()
        self.executors: Dict[RunIdentifier, Executor] = {}
        self.thread_pool = ThreadPoolExecutor(5)
        self.futures = []
        self._manager = mp.Manager()
        # Cannot type-hint more specifically than DictProxy,
        # but this functions as a Dict[RunIdentifier, RunStatus]
        self.run_statuses: DictProxy = self._manager.dict()

        configure_session(api_key="prompt", cache="yes", validate=True)
        # TODO: Only create the project here if using SDK; if using Trainer, we'll make separate projects for each run
        self._project = create_project(display_name=self.config.project_display_name)

    def execute(self):
        for dataset in self.datasets:
            for model in self.gretel_models:
                run_identifier = (dataset.name, model().name)
                logger.info(f"Queueing run `{run_identifier}`")
                # if self.config.trainer:
                #     executor = GretelTrainerExecutor(...)
                # else:
                #     exector = GretelSDKExecutor(...)
                executor = GretelSDKExecutor(
                    project=self._project,
                    benchmark_model=model(),
                    run_identifier=run_identifier,
                    statuses=self.run_statuses,
                    refresh_interval=self.config.refresh_interval,
                )
                self.executors[run_identifier] = executor
                self.futures.append(self.thread_pool.submit(_run, executor, dataset))
            for model in self.custom_models:
                pass

    @property
    def results(self) -> pd.DataFrame:
        pass

    def wait(self):
        [future.result() for future in self.futures]
        return self


def _run(executor: GretelSDKExecutor, dataset: Dataset) -> None:
    executor.train(dataset)
    executor.generate()
