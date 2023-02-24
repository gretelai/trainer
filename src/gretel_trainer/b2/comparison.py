from __future__ import annotations

import logging
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.managers import DictProxy
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import pandas as pd
from gretel_client import configure_session
from gretel_client.projects import create_project

from gretel_trainer.b2.core import BenchmarkConfig, Dataset, RunIdentifier
from gretel_trainer.b2.custom_executor import CustomExecutor
from gretel_trainer.b2.custom_models import CustomModel
from gretel_trainer.b2.gretel_executor import GretelExecutor
from gretel_trainer.b2.gretel_models import GretelModel
from gretel_trainer.b2.status import Completed, Failed, InProgress

logger = logging.getLogger(__name__)


ModelTypes = Union[Type[CustomModel], Type[GretelModel]]
Executor = Union[GretelExecutor, CustomExecutor]


class Comparison:
    def __init__(
        self,
        *,
        datasets: List[Dataset],
        models: List[ModelTypes],
        config: BenchmarkConfig,
    ):
        self.datasets = datasets
        self.gretel_models = [m for m in models if issubclass(m, GretelModel)]
        self.custom_models = [m for m in models if not m in self.gretel_models]
        self.config = config
        self.executors: Dict[RunIdentifier, Executor] = {}
        self.thread_pool = ThreadPoolExecutor(5)
        self.futures = []
        self._manager = mp.Manager()
        # Cannot type-hint more specifically than DictProxy,
        # but this functions as a Dict[RunIdentifier, RunStatus]
        self.run_statuses: DictProxy = self._manager.dict()

        configure_session(api_key="prompt", cache="yes", validate=True)
        self._project = None
        if not self.config.trainer:
            self._project = create_project(
                display_name=self.config.project_display_name
            )

    def execute(self) -> Comparison:
        custom_executors: List[CustomExecutor] = []
        for dataset in self.datasets:
            for model in self.gretel_models:
                self._setup_gretel_run(dataset, model)
            for model_type in self.custom_models:
                self._setup_custom_run(
                    dataset, model_type, custom_executors
                )  # type:ignore
        self.futures.append(self.thread_pool.submit(_run_custom, custom_executors))
        return self

    @property
    def results(self) -> pd.DataFrame:
        result_records = [self._result_dict(run_id) for run_id in self.executors]
        return pd.DataFrame.from_records(result_records)

    def wait(self) -> Comparison:
        [future.result() for future in self.futures]
        return self

    def _result_dict(self, run_identifier: RunIdentifier) -> Dict[str, Any]:
        executor = self.executors[run_identifier]
        status = self.run_statuses[run_identifier]

        sqs = None
        if isinstance(status, Completed):
            sqs = status.sqs

        train_time = None
        if isinstance(status, (Completed, Failed, InProgress)):
            train_time = status.train_secs

        generate_time = None
        if isinstance(status, (Completed, Failed, InProgress)):
            generate_time = status.generate_secs

        total_time = train_time
        if train_time is not None and generate_time is not None:
            total_time = train_time + generate_time

        return {
            "Input data": executor.dataset.name,
            "Model": executor.model_name,
            "DataType": executor.dataset.datatype,
            "Rows": executor.dataset.row_count,
            "Columns": executor.dataset.column_count,
            "Status": status.display,
            "SQS": sqs,
            "Train time (sec)": train_time,
            "Generate time (sec)": generate_time,
            "Total time (sec)": total_time,
        }

    def _setup_gretel_run(self, dataset: Dataset, model: Type[GretelModel]) -> None:
        run_identifier = RunIdentifier((dataset.name, model().name))
        logger.info(f"Queueing run `{run_identifier}`")
        executor = GretelExecutor(
            benchmark_model=model(),
            dataset=dataset,
            run_identifier=run_identifier,
            statuses=self.run_statuses,
            config=self.config,
            project=self._project,
        )
        self.executors[run_identifier] = executor
        self.futures.append(self.thread_pool.submit(_run_gretel, executor))

    def _setup_custom_run(
        self,
        dataset: Dataset,
        model_type: Type[CustomModel],
        collection: List[CustomExecutor],
    ) -> None:
        run_identifier = RunIdentifier((dataset.name, model_type.__name__))
        executor = CustomExecutor(
            model=model_type(),
            dataset=dataset,
            run_identifier=run_identifier,
            statuses=self.run_statuses,
        )
        self.executors[run_identifier] = executor
        collection.append(executor)


def _run_gretel(executor: GretelExecutor) -> None:
    executor.train()
    executor.generate()


def _run_custom(executors: List[CustomExecutor]) -> None:
    for executor in executors:
        executor.train()
        executor.generate()
