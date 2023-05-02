from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from inspect import isclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import pandas as pd
from gretel_client import configure_session
from gretel_client.helpers import poll
from gretel_client.projects import Project, create_project, search_projects
from gretel_client.projects.jobs import Job

from gretel_trainer.benchmark.core import (
    BenchmarkConfig,
    BenchmarkException,
    Dataset,
    log,
)
from gretel_trainer.benchmark.custom.datasets import CustomDataset
from gretel_trainer.benchmark.custom.models import CustomModel
from gretel_trainer.benchmark.custom.strategy import CustomStrategy
from gretel_trainer.benchmark.executor import Executor
from gretel_trainer.benchmark.gretel.datasets import GretelDataset
from gretel_trainer.benchmark.gretel.models import GretelAuto, GretelModel
from gretel_trainer.benchmark.gretel.strategy_sdk import GretelSDKStrategy
from gretel_trainer.benchmark.gretel.strategy_trainer import GretelTrainerStrategy

logger = logging.getLogger(__name__)

ALL_REPORT_SCORES = {
    "SQS": "synthetic_data_quality_score",
    "FCS": "field_correlation_stability",
    "PCS": "principal_component_stability",
    "FDS": "field_distribution_stability",
    "PPL": "privacy_protection_level",
}


DatasetTypes = Union[CustomDataset, GretelDataset]
ModelTypes = Union[
    CustomModel,
    Type[CustomModel],
    GretelModel,
    Type[GretelModel],
]


class RunKey(Tuple[str, str]):
    RUN_IDENTIFIER_SEPARATOR = "-"

    @property
    def model_name(self) -> str:
        return self[0]

    @property
    def dataset_name(self) -> str:
        return self[1]

    @property
    def identifier(self) -> str:
        """Identifier returns a flat string identifying this run.

        Returns:
            A string identifying the run.
        """
        return self.RUN_IDENTIFIER_SEPARATOR.join(self)

    def __repr__(self) -> str:
        return f"({self.model_name}, {self.dataset_name})"


FutureKeyT = Union[str, RunKey]


def compare(
    *,
    datasets: List[DatasetTypes],
    models: List[ModelTypes],
    project_display_name: Optional[str] = None,
    trainer: bool = False,
    refresh_interval: int = 15,
    working_dir: Optional[str] = None,
    additional_report_scores: Optional[List[str]] = None,
) -> Comparison:
    comparison = Comparison(
        datasets=datasets,
        models=models,
        project_display_name=project_display_name,
        trainer=trainer,
        refresh_interval=refresh_interval,
        working_dir=working_dir,
        additional_report_scores=additional_report_scores,
    )
    return comparison.prepare().execute()


class Comparison:
    def __init__(
        self,
        *,
        datasets: List[DatasetTypes],
        models: List[ModelTypes],
        project_display_name: Optional[str] = None,
        trainer: bool = False,
        refresh_interval: int = 15,
        working_dir: Optional[str] = None,
        additional_report_scores: Optional[List[str]] = None,
    ):
        model_instances = [
            cast(Union[GretelModel, CustomModel], m() if isclass(m) else m)
            for m in models
        ]
        self.gretel_models = [m for m in model_instances if isinstance(m, GretelModel)]
        self.custom_models = [
            m for m in model_instances if not isinstance(m, GretelModel)
        ]

        project_display_name = project_display_name or _default_name()
        working_dir = working_dir or project_display_name
        self.config = BenchmarkConfig(
            project_display_name=project_display_name,
            trainer=trainer,
            refresh_interval=refresh_interval,
            working_dir=Path(working_dir),
        )

        _validate_setup(self.config, self.gretel_models, self.custom_models, datasets)

        configure_session(api_key="prompt", cache="yes", validate=True)

        self.config.working_dir.mkdir(exist_ok=True)
        self.datasets = [
            _make_dataset(dataset, self.config.working_dir) for dataset in datasets
        ]
        self._gretel_executors: Dict[RunKey, Executor] = {}
        self._custom_executors: Dict[RunKey, Executor] = {}
        self.trainer_project_names: Dict[str, str] = {}
        self.thread_pool = ThreadPoolExecutor(5)
        self.futures: Dict[FutureKeyT, Future] = {}

        self._report_scores = {
            score_name: ALL_REPORT_SCORES[score_name]
            for score_name in ["SQS"] + (additional_report_scores or [])
        }

    @property
    def executors(self) -> Dict[RunKey, Executor]:
        return self._gretel_executors | self._custom_executors

    def _make_project(self) -> Project:
        display_name = self.config.project_display_name
        if self.config.trainer:
            display_name = f"{display_name}-evaluate"

        return create_project(display_name=display_name)

    def prepare(self) -> Comparison:
        self._project = self._make_project()

        _trainer_project_index = 0
        for dataset in self.datasets:
            artifact_key = (
                _upload_dataset_to_project(
                    dataset.data_source, self._project, self.config.trainer
                )
                if not dataset.public
                else None
            )

            for model in self.gretel_models:
                self._setup_gretel_run(
                    dataset, model, artifact_key, _trainer_project_index
                )
                _trainer_project_index += 1

            for model in self.custom_models:
                self._setup_custom_run(dataset, model, artifact_key)

        return self

    def execute(self) -> Comparison:
        for run_key, executor in self._gretel_executors.items():
            self.futures[run_key] = self.thread_pool.submit(_run_gretel, executor)

        if len(self._custom_executors) > 0:
            self.futures["custom"] = self.thread_pool.submit(
                _run_custom, list(self._custom_executors.values())
            )

        return self

    @property
    def results(self) -> pd.DataFrame:
        result_records = [self._result_dict(run_key) for run_key in self.executors]
        return pd.DataFrame.from_records(result_records)

    def export_results(self, destination: str) -> None:
        self.results.to_csv(destination, index=False)

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
        run_key = RunKey((model, dataset))
        executor = self.executors[run_key]
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

    def _result_dict(self, run_key: RunKey) -> Dict[str, Any]:
        executor = self.executors[run_key]
        model_name, dataset_name = run_key

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
            **{
                score_name: executor.get_report_score(score_key)
                for score_name, score_key in self._report_scores.items()
            },
            "Train time (sec)": train_time,
            "Generate time (sec)": generate_time,
            "Total time (sec)": total_time,
        }

    def _setup_gretel_run(
        self,
        dataset: Dataset,
        model: GretelModel,
        artifact_key: Optional[str],
        trainer_project_index: int,
    ) -> None:
        run_key = _make_run_key(model, dataset)
        run_identifier = run_key.identifier
        strategy = self._set_gretel_strategy(
            benchmark_model=model,
            dataset=dataset,
            run_identifier=run_identifier,
            trainer_project_index=trainer_project_index,
            artifact_key=artifact_key,
        )
        executor = Executor(
            strategy=strategy,
            run_identifier=run_identifier,
            evaluate_project=self._project,
            config=self.config,
        )
        self._gretel_executors[run_key] = executor

    def _setup_custom_run(
        self,
        dataset: Dataset,
        model: CustomModel,
        artifact_key: Optional[str] = None,
    ) -> None:
        run_key = _make_run_key(model, dataset)
        run_identifier = run_key.identifier
        strategy = CustomStrategy(
            benchmark_model=model,
            dataset=dataset,
            run_identifier=run_identifier,
            config=self.config,
            artifact_key=artifact_key,
        )
        executor = Executor(
            strategy=strategy,
            run_identifier=run_identifier,
            evaluate_project=self._project,
            config=self.config,
        )
        self._custom_executors[run_key] = executor

    def _basic_status(self, future: Future) -> str:
        if future.done():
            return "Finished"
        elif future.cancelled():
            return "Cancelled"
        else:
            return "Running"

    def _set_gretel_strategy(
        self,
        benchmark_model: GretelModel,
        dataset: Dataset,
        run_identifier: str,
        trainer_project_index: int,
        artifact_key: Optional[str],
    ) -> Union[GretelSDKStrategy, GretelTrainerStrategy]:
        if self.config.trainer:
            trainer_project_name = _trainer_project_name(
                self.config, trainer_project_index
            )
            self.trainer_project_names[run_identifier] = trainer_project_name
            return GretelTrainerStrategy(
                benchmark_model=benchmark_model,
                dataset=dataset,
                run_identifier=run_identifier,
                project_name=trainer_project_name,
                config=self.config,
            )
        else:
            return GretelSDKStrategy(
                benchmark_model=benchmark_model,
                dataset=dataset,
                artifact_key=artifact_key,
                run_identifier=run_identifier,
                project=self._project,
                config=self.config,
            )


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
        public=dataset.public,
    )


def _make_run_key(model: Union[GretelModel, CustomModel], dataset: Dataset) -> RunKey:
    return RunKey((_model_name(model), dataset.name))


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


def _model_name(model: Union[GretelModel, CustomModel]) -> str:
    if isinstance(model, GretelModel):
        return model.name
    else:
        return type(model).__name__


def _default_name() -> str:
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"benchmark-{current_time}"


def _trainer_project_name(config: BenchmarkConfig, index: int) -> str:
    prefix = config.project_display_name
    name = f"{prefix}-{index}"
    return name.replace("_", "-")


def _upload_dataset_to_project(
    source: str, project: Project, trainer: bool
) -> Optional[str]:
    if trainer:
        return None

    return project.upload_artifact(source)
