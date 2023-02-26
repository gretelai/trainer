from __future__ import annotations

import logging
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from multiprocessing.managers import DictProxy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import pandas as pd
from gretel_client import configure_session
from gretel_client.projects import create_project

from gretel_trainer.b2.core import (
    BenchmarkConfig,
    BenchmarkException,
    Dataset,
    RunIdentifier,
)
from gretel_trainer.b2.custom_datasets import CustomDataset
from gretel_trainer.b2.custom_executor import CustomExecutor
from gretel_trainer.b2.custom_models import CustomModel
from gretel_trainer.b2.gretel_datasets import GretelDataset
from gretel_trainer.b2.gretel_executor import GretelExecutor
from gretel_trainer.b2.gretel_models import GretelModel
from gretel_trainer.b2.status import Completed, Failed, InProgress

logger = logging.getLogger(__name__)


DatasetTypes = Union[CustomDataset, GretelDataset]
ModelTypes = Union[Type[CustomModel], Type[GretelModel]]
Executor = Union[CustomExecutor, GretelExecutor]


def compare(
    *,
    datasets: List[DatasetTypes],
    models: List[ModelTypes],
    project_display_name: str = "benchmark",
    trainer: bool = False,
    refresh_interval: int = 15,
    working_dir: str = "benchmark",
) -> Comparison:
    config = BenchmarkConfig(
        project_display_name=project_display_name,
        trainer=trainer,
        refresh_interval=refresh_interval,
        working_dir=Path(working_dir),
        timestamp=_current_timestamp(),
    )
    comparison = Comparison(
        datasets=datasets,
        models=models,
        config=config,
    )
    return comparison.execute()


class Comparison:
    def __init__(
        self,
        *,
        datasets: List[DatasetTypes],
        models: List[ModelTypes],
        config: BenchmarkConfig,
    ):
        self.gretel_models = [m() for m in models if issubclass(m, GretelModel)]
        self.custom_models = [m for m in models if not issubclass(m, GretelModel)]
        self.config = config
        _validate_setup(self.config, self.gretel_models)
        self.config.working_dir.mkdir(exist_ok=True)
        self.datasets = [
            _make_dataset(dataset, self.config.working_dir) for dataset in datasets
        ]
        self.executors: Dict[RunIdentifier, Executor] = {}
        self.thread_pool = ThreadPoolExecutor(5)
        self.futures = []
        self._manager = mp.Manager()
        # Cannot type-hint more specifically than DictProxy,
        # but this functions as a Dict[RunIdentifier, RunStatus]
        self.run_statuses: DictProxy = self._manager.dict()

        configure_session(api_key="prompt", cache="yes", validate=True)
        self._project = None
        if not self.config.trainer and len(self.gretel_models) > 0:
            self._project = create_project(
                display_name=self.config.project_display_name
            )

    def execute(self) -> Comparison:
        custom_executors: List[CustomExecutor] = []
        for dataset in self.datasets:
            for model in self.gretel_models:
                self._setup_gretel_run(dataset, model)
            for model_type in self.custom_models:
                self._setup_custom_run(dataset, model_type, custom_executors)
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

    def _setup_gretel_run(self, dataset: Dataset, model: GretelModel) -> None:
        run_identifier = RunIdentifier((dataset.name, model.name))
        logger.info(f"Queueing run `{run_identifier}`")
        executor = GretelExecutor(
            benchmark_model=model,
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
        model_name = model_type.__name__
        run_identifier = RunIdentifier((dataset.name, model_name))
        logger.info(f"Queueing run `{run_identifier}`")
        executor = CustomExecutor(
            model=model_type(),
            model_name=model_name,
            dataset=dataset,
            run_identifier=run_identifier,
            working_dir=self.config.working_dir,
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


def _make_dataset(dataset: DatasetTypes, working_dir: Path) -> Dataset:
    source = dataset.data_source
    if isinstance(source, pd.DataFrame):
        csv_path = working_dir / f"{dataset.name}.csv"
        source.to_csv(csv_path, index=False)
        source = str(csv_path)

    return Dataset(
        name=dataset.name,
        datatype=dataset.datatype,
        row_count=dataset.row_count,
        column_count=dataset.column_count,
        data_source=source,
    )


def _current_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")


def _validate_setup(config: BenchmarkConfig, gretel_models: List[GretelModel]) -> None:
    if config.trainer:
        unsupported_models = []
        for model in gretel_models:
            if model.trainer_model_type is None:
                logger.error(
                    f"Model `{model.name}` (model key `{model.model_key}`) is not supported by Trainer. "
                    "Either remove it from this comparison, or configure this comparison to use the SDK (trainer=False)"
                )
                unsupported_models.append(model)
        if len(unsupported_models) > 0:
            raise BenchmarkException("Invalid configuration")
