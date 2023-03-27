from __future__ import annotations

import logging
import multiprocessing as mp
from concurrent.futures import Future, ThreadPoolExecutor
from multiprocessing.managers import DictProxy
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type, Union

import pandas as pd
from gretel_client import configure_session
from gretel_client.helpers import poll
from gretel_client.projects import Project, create_project, search_projects
from gretel_client.projects.jobs import Job

from gretel_trainer.b2.core import BenchmarkConfig, BenchmarkException, Dataset, log
from gretel_trainer.b2.custom.datasets import CustomDataset
from gretel_trainer.b2.custom.models import CustomModel
from gretel_trainer.b2.custom.strategy import CustomStrategy
from gretel_trainer.b2.executor import Executor
from gretel_trainer.b2.gretel.datasets import GretelDataset
from gretel_trainer.b2.gretel.models import GretelAuto, GretelModel
from gretel_trainer.b2.gretel.strategy_sdk import GretelSDKStrategy
from gretel_trainer.b2.gretel.strategy_trainer import GretelTrainerStrategy

logger = logging.getLogger(__name__)

RUN_IDENTIFIER_SEPARATOR = "-"


DatasetTypes = Union[CustomDataset, GretelDataset]
ModelTypes = Union[Type[CustomModel], Type[GretelModel]]


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
        self.custom_models = [m() for m in models if not issubclass(m, GretelModel)]
        self.config = config
        _validate_setup(self.config, self.gretel_models, self.custom_models, datasets)

        configure_session(api_key="prompt", cache="yes", validate=True)
        self._project = self._make_project()

        self.config.working_dir.mkdir(exist_ok=True)
        self.datasets = [
            _make_dataset(dataset, self.config.working_dir) for dataset in datasets
        ]
        self.executors: Dict[str, Executor] = {}
        self.thread_pool = ThreadPoolExecutor(5)
        self.futures: Dict[str, Future] = {}

    def _make_project(self) -> Project:
        display_name = self.config.project_display_name
        if self.config.trainer:
            display_name = f"{display_name}-evaluate"

        return create_project(display_name=display_name)

    def execute(self) -> Comparison:
        custom_executors: List[Executor] = []
        for dataset in self.datasets:
            for model in self.gretel_models:
                self._setup_gretel_run(dataset, model)
            for model in self.custom_models:
                self._setup_custom_run(dataset, model, custom_executors)
        if len(custom_executors) > 0:
            self.futures["custom"] = self.thread_pool.submit(
                _run_custom, custom_executors
            )
        return self

    @property
    def results(self) -> pd.DataFrame:
        result_records = [self._result_dict(run_id) for run_id in self.executors]
        return pd.DataFrame.from_records(result_records)

    def wait(self) -> Comparison:
        [future.result() for future in self.futures.values()]
        return self

    def whats_happening(self) -> Dict[str, str]:
        return {
            str(key): self._basic_status(future) for key, future in self.futures.items()
        }

    def poll_job(self, model: str, dataset: str, action: str):
        job = self._get_gretel_job(model, dataset, action)
        poll(job)

    def get_logs(self, model: str, dataset: str, action: str):
        job = self._get_gretel_job(model, dataset, action)
        return job.logs

    def _get_gretel_job(self, model: str, dataset: str, action: str) -> Job:
        run_id = f"{model}-{dataset}"
        executor = self.executors[run_id]
        strategy = executor.strategy
        if not isinstance(strategy, GretelSDKStrategy):
            raise BenchmarkException("Cannot get Gretel job for non-GretelSDK runs")

        if action == "train":
            job = strategy.model
        elif action == "generate":
            job = strategy.record_handler
        else:
            raise BenchmarkException(
                f"Unrecognized action `{action}` (must be `train` or `generate`)"
            )

        if job is None:
            raise BenchmarkException("Gretel Job does not exist")

        return job

    def cleanup(self) -> None:
        self._project.delete()

        if self.config.trainer:
            for project in search_projects(query=self.config.project_display_name):
                project.delete()

    def _result_dict(self, run_identifier: str) -> Dict[str, Any]:
        executor = self.executors[run_identifier]
        model_name, dataset_name = run_identifier.split(RUN_IDENTIFIER_SEPARATOR, 1)

        train_time = executor.strategy.get_train_time()
        generate_time = executor.strategy.get_generate_time()

        total_time = train_time
        if train_time is not None and generate_time is not None:
            total_time = train_time + generate_time

        return {
            "Input data": dataset_name,
            "Model": model_name,
            "DataType": executor.strategy.dataset.datatype,
            "Rows": executor.strategy.dataset.row_count,
            "Columns": executor.strategy.dataset.column_count,
            "Status": executor.status.value,
            "SQS": executor.get_sqs_score(),
            "Train time (sec)": train_time,
            "Generate time (sec)": generate_time,
            "Total time (sec)": total_time,
        }

    def _setup_gretel_run(self, dataset: Dataset, model: GretelModel) -> None:
        run_identifier = _make_run_identifier(model, dataset)
        strategy = _set_gretel_strategy(
            benchmark_model=model,
            dataset=dataset,
            run_identifier=run_identifier,
            config=self.config,
            project=self._project,
        )
        executor = Executor(
            strategy=strategy,
            run_identifier=run_identifier,
            evaluate_project=self._project,
            config=self.config,
        )
        self.executors[run_identifier] = executor
        self.futures[run_identifier] = self.thread_pool.submit(_run_gretel, executor)

    def _setup_custom_run(
        self,
        dataset: Dataset,
        model: CustomModel,
        collection: List[Executor],
    ) -> None:
        run_identifier = _make_run_identifier(model, dataset)
        strategy = CustomStrategy(
            benchmark_model=model,
            dataset=dataset,
            run_identifier=run_identifier,
            config=self.config,
        )
        executor = Executor(
            strategy=strategy,
            run_identifier=run_identifier,
            evaluate_project=self._project,
            config=self.config,
        )
        self.executors[run_identifier] = executor
        collection.append(executor)

    def _basic_status(self, future: Future) -> str:
        if future.done():
            return "Finished"
        elif future.cancelled():
            return "Cancelled"
        else:
            return "Running"


def _run_gretel(executor: Executor) -> None:
    executor.run()


def _run_custom(executors: List[Executor]) -> None:
    for executor in executors:
        executor.run()


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


def _make_run_identifier(
    model: Union[GretelModel, CustomModel], dataset: Dataset
) -> str:
    run_identifier = f"{_model_name(model)}{RUN_IDENTIFIER_SEPARATOR}{dataset.name}"
    log(run_identifier, "preparing run")
    return run_identifier


def _validate_setup(
    config: BenchmarkConfig,
    gretel_models: List[GretelModel],
    custom_models: List[CustomModel],
    all_datasets: List[DatasetTypes],
) -> None:
    dataset_names = [d.name for d in all_datasets]
    model_names = [_model_name(m) for m in (gretel_models + custom_models)]
    _ensure_unique(dataset_names, "datasets")
    _ensure_unique(model_names, "models")

    if config.trainer:
        _validate_trainer_setup(gretel_models)
    else:
        _validate_sdk_setup(gretel_models)


def _ensure_unique(col: List[str], kind: str) -> None:
    if len(set(col)) < len(col):
        raise BenchmarkException(f"{kind} must have unique names")


def _validate_trainer_setup(gretel_models: List[GretelModel]) -> None:
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


def _validate_sdk_setup(gretel_models: List[GretelModel]) -> None:
    if any(isinstance(m, GretelAuto) for m in gretel_models):
        logger.error(
            "GretelAuto is only supported when using Trainer. "
            "Either remove it from this comparison, or configure this comparison to use Trainer (trainer=True)"
        )
        raise BenchmarkException("Invalid configuration")


def _set_gretel_strategy(
    benchmark_model: GretelModel,
    dataset: Dataset,
    run_identifier: str,
    config: BenchmarkConfig,
    project: Project,
) -> Union[GretelSDKStrategy, GretelTrainerStrategy]:
    if config.trainer:
        return GretelTrainerStrategy(
            benchmark_model=benchmark_model,
            dataset=dataset,
            run_identifier=run_identifier,
            config=config,
        )
    else:
        return GretelSDKStrategy(
            benchmark_model=benchmark_model,
            dataset=dataset,
            run_identifier=run_identifier,
            project=project,
            config=config,
        )


def _model_name(model: Union[GretelModel, CustomModel]) -> str:
    if isinstance(model, GretelModel):
        return model.name
    else:
        return type(model).__name__
