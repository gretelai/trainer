from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Optional, Union

import pandas as pd
from gretel_client.helpers import poll
from gretel_client.projects import Project, create_project, search_projects
from gretel_client.projects.jobs import Job
from typing_extensions import TypeGuard

from gretel_trainer.benchmark.core import BenchmarkConfig, BenchmarkException
from gretel_trainer.benchmark.custom.models import CustomModel
from gretel_trainer.benchmark.custom.strategy import CustomStrategy
from gretel_trainer.benchmark.executor import Executor
from gretel_trainer.benchmark.gretel.models import GretelAuto, GretelModel
from gretel_trainer.benchmark.gretel.strategy_sdk import GretelSDKStrategy
from gretel_trainer.benchmark.gretel.strategy_trainer import GretelTrainerStrategy
from gretel_trainer.benchmark.job_spec import AnyModelType, JobSpec, RunKey

logger = logging.getLogger(__name__)

ALL_REPORT_SCORES = {
    "SQS": "synthetic_data_quality_score",
    "FCS": "field_correlation_stability",
    "PCS": "principal_component_stability",
    "FDS": "field_distribution_stability",
    "PPL": "privacy_protection_level",
}

FutureKeyT = Union[str, RunKey]


class Session:
    def __init__(
        self,
        *,
        jobs: list[JobSpec[AnyModelType]],
        config: Optional[BenchmarkConfig] = None,
    ):
        self._jobs = jobs
        self._config = config or BenchmarkConfig()

        _validate_jobs(self._config, jobs)

        self._gretel_executors: dict[RunKey, Executor] = {}
        self._custom_executors: dict[RunKey, Executor] = {}
        self._trainer_project_names: dict[str, str] = {}
        self._thread_pool = ThreadPoolExecutor(5)
        self._futures: dict[FutureKeyT, Future] = {}

        self._report_scores = {
            score_name: ALL_REPORT_SCORES[score_name]
            for score_name in ["SQS"] + self._config.additional_report_scores
        }

    @property
    def executors(self) -> dict[RunKey, Executor]:
        return self._gretel_executors | self._custom_executors

    def _make_project(self) -> Project:
        display_name = self._config.project_display_name
        if self._config.trainer:
            display_name = f"{display_name}-evaluate"

        return create_project(display_name=display_name)

    def prepare(self) -> Session:
        self._project = self._make_project()

        _trainer_project_index = 0

        data_source_map = {}
        for job in self._jobs:
            if (artifact_key := data_source_map.get(job.dataset.data_source)) is None:
                # TODO: avoid upload for public datasets, when possible
                artifact_key = _upload_dataset_to_project(
                    job.dataset.data_source,
                    self._project,
                    self._config.trainer,
                )
                data_source_map[job.dataset.data_source] = artifact_key

            if is_gretel_model(job):
                self._setup_gretel_run(job, artifact_key, _trainer_project_index)
                _trainer_project_index += 1

            elif is_custom_model(job):
                self._setup_custom_run(job, artifact_key)

            else:
                raise BenchmarkException(
                    f"Unexpected model class received: {job.model.__class__.__name__}!"
                )

        return self

    def execute(self) -> Session:
        for run_key, executor in self._gretel_executors.items():
            self._futures[run_key] = self._thread_pool.submit(_run_gretel, executor)

        if len(self._custom_executors) > 0:
            self._futures["custom"] = self._thread_pool.submit(
                _run_custom, list(self._custom_executors.values())
            )

        return self

    @property
    def results(self) -> pd.DataFrame:
        result_records = [self._result_dict(run_key) for run_key in self.executors]
        return pd.DataFrame.from_records(result_records)

    def export_results(self, destination: str) -> None:
        self.results.to_csv(destination, index=False)

    def wait(self) -> Session:
        [future.result() for future in self._futures.values()]
        return self

    def whats_happening(self) -> dict[str, str]:
        return {
            str(key): self._basic_status(future)
            for key, future in self._futures.items()
        }

    def poll_job(self, model: str, dataset: str, action: str) -> None:
        job = self._get_gretel_job(model, dataset, action)
        poll(job)

    def get_logs(self, model: str, dataset: str, action: str) -> list:
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

        if self._config.trainer:
            for project in search_projects(query=self._config.project_display_name):
                project.delete()

    def _result_dict(self, run_key: RunKey) -> dict[str, Any]:
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
        job: JobSpec[GretelModel],
        artifact_key: Optional[str],
        trainer_project_index: int,
    ) -> None:
        run_key = job.make_run_key()
        run_identifier = run_key.identifier
        strategy = self._create_gretel_strategy(
            job=job,
            run_identifier=run_identifier,
            trainer_project_index=trainer_project_index,
            artifact_key=artifact_key,
        )
        executor = Executor(
            strategy=strategy,
            run_identifier=run_identifier,
            evaluate_project=self._project,
            config=self._config,
        )
        self._gretel_executors[run_key] = executor

    def _setup_custom_run(
        self,
        job: JobSpec[CustomModel],
        artifact_key: Optional[str] = None,
    ) -> None:
        run_key = job.make_run_key()
        run_identifier = run_key.identifier
        strategy = CustomStrategy(
            benchmark_model=job.model,
            dataset=job.dataset,
            run_identifier=run_identifier,
            config=self._config,
            artifact_key=artifact_key,
        )
        executor = Executor(
            strategy=strategy,
            run_identifier=run_identifier,
            evaluate_project=self._project,
            config=self._config,
        )
        self._custom_executors[run_key] = executor

    def _basic_status(self, future: Future) -> str:
        if future.done():
            return "Finished"
        elif future.cancelled():
            return "Cancelled"
        else:
            return "Running"

    def _create_gretel_strategy(
        self,
        job: JobSpec[GretelModel],
        run_identifier: str,
        trainer_project_index: int,
        artifact_key: Optional[str],
    ) -> Union[GretelSDKStrategy, GretelTrainerStrategy]:
        if self._config.trainer:
            trainer_project_name = _trainer_project_name(
                self._config, trainer_project_index
            )
            self._trainer_project_names[run_identifier] = trainer_project_name
            return GretelTrainerStrategy(
                job_spec=job,
                run_identifier=run_identifier,
                project_name=trainer_project_name,
                config=self._config,
            )
        else:
            return GretelSDKStrategy(
                job_spec=job,
                artifact_key=artifact_key,
                run_identifier=run_identifier,
                project=self._project,
                config=self._config,
            )


def is_gretel_model(job: JobSpec[AnyModelType]) -> TypeGuard[JobSpec[GretelModel]]:
    return isinstance(job.model, GretelModel)


def is_custom_model(job: JobSpec[AnyModelType]) -> TypeGuard[JobSpec[CustomModel]]:
    return not isinstance(job.model, GretelModel)


def _run_gretel(executor: Executor) -> None:
    executor.run()


def _run_custom(executors: list[Executor]) -> None:
    for executor in executors:
        executor.run()


def _validate_jobs(config: BenchmarkConfig, jobs: list[JobSpec[AnyModelType]]) -> None:
    gretel_models = [j.model for j in jobs if is_gretel_model(j)]
    if config.trainer:
        _validate_trainer_setup(gretel_models)
    else:
        _validate_sdk_setup(gretel_models)


def _validate_trainer_setup(gretel_models: list[GretelModel]) -> None:
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


def _validate_sdk_setup(gretel_models: list[GretelModel]) -> None:
    if any(isinstance(m, GretelAuto) for m in gretel_models):
        logger.error(
            "GretelAuto is only supported when using Trainer. "
            "Either remove it from this comparison, or configure this comparison to use Trainer (trainer=True)"
        )
        raise BenchmarkException("Invalid configuration")


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
