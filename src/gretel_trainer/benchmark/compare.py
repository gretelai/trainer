import concurrent.futures as futures
import multiprocessing as mp
import queue
import shutil
import time

from contextlib import suppress
from dataclasses import dataclass
from typing import Callable, Dict, List, Type, Union

import pandas as pd

from gretel_trainer.benchmark.core import (
    BenchmarkException,
    Completed,
    Dataset,
    Evaluator,
    execute,
    Failed,
    InProgress,
    ModelFactory,
    NotStarted,
    Run,
    Skipped,
)
from gretel_trainer.benchmark.custom.models import CustomExecutor
from gretel_trainer.benchmark.gretel.models import GretelModel
from gretel_trainer.benchmark.gretel.sdk import GretelSDK, GretelSDKExecutor
from gretel_trainer.benchmark.gretel.trainer import GretelTrainerExecutor, Trainer

from gretel_client.projects.exceptions import ModelConfigError
from gretel_client.projects.models import read_model_config

GretelExecutor = Union[GretelSDKExecutor, GretelTrainerExecutor]


@dataclass
class RuntimeConfig:
    cleanup_wait_secs: float
    local_dir: str
    project_prefix: Callable[[], str]
    thread_pool: futures.Executor


class Comparison:
    def __init__(
        self,
        gretel_model_runs: List[Run[GretelExecutor]],
        custom_model_runs: List[Run[CustomExecutor]],
        runtime_config: RuntimeConfig,
    ):
        self.gretel_model_runs = gretel_model_runs
        self.custom_model_runs = custom_model_runs
        self.runtime_config = runtime_config
        self.futures = []
        self.results_queue = mp.Queue()

    @property
    def all_runs(self):
        return self.gretel_model_runs + self.custom_model_runs

    @property
    def is_complete(self):
        return all(
            isinstance(run.status, (Completed, Failed, Skipped))
            for run in self.all_runs
        )

    @property
    def results(self) -> pd.DataFrame:
        while True:
            try:
                result = self.results_queue.get_nowait()
                for run in self.all_runs:
                    if run.identifier == result["identifier"]:
                        run.status = result["status"]
            except queue.Empty:
                break

        result_dicts = [_result_dict(run) for run in self.all_runs]
        return pd.DataFrame.from_records(result_dicts)

    def export_results(self, destination: str):
        self.results.to_csv(destination, index=False)

    def execute(self):
        parallel_runs_futures = [
            self.runtime_config.thread_pool.submit(execute, run, self.results_queue)
            for run in self.gretel_model_runs
        ]
        self.futures.extend(parallel_runs_futures)

        _run_sequentially(self.custom_model_runs, self.results_queue)

        cleanup_future = self.runtime_config.thread_pool.submit(self._cleanup)
        self.futures.append(cleanup_future)

        return self

    def wait(self):
        [future.result() for future in self.futures]
        return self

    def _cleanup(self):
        while not self.is_complete:
            time.sleep(self.runtime_config.cleanup_wait_secs)
            self.results

        with suppress(FileNotFoundError):
            shutil.rmtree(self.runtime_config.local_dir)


def _run_sequentially(runs, queue):
    for run in runs:
        execute(run, queue)


def _result_dict(run: Run) -> Dict:
    sqs = None
    if isinstance(run.status, Completed):
        sqs = run.status.sqs

    train_time = None
    if isinstance(run.status, (Completed, Failed, InProgress)):
        train_time = run.status.train_secs

    generate_time = None
    if isinstance(run.status, (Completed, Failed, InProgress)):
        generate_time = run.status.generate_secs

    total_time = train_time
    if train_time is not None and generate_time is not None:
        total_time = train_time + generate_time

    return {
        "Input data": run.source.name,
        "Model": run.executor.model_name,
        "DataType": run.source.datatype,
        "Rows": run.source.row_count,
        "Columns": run.source.column_count,
        "Status": run.status.display,
        "SQS": sqs,
        "Train time": train_time,
        "Generate time": generate_time,
        "Total time": total_time,
    }


def compare(
    *,
    datasets: List[Dataset],
    models: List[Union[ModelFactory, Type[GretelModel]]],
    runtime_config: RuntimeConfig,
    gretel_sdk: GretelSDK,
    evaluator: Evaluator,
    gretel_trainer_factory: Callable[..., Trainer],
) -> Comparison:
    project_name_prefix = f"benchmark-{runtime_config.project_prefix()}"

    gretel_model_runs: List[Run[GretelExecutor]] = []
    custom_model_runs: List[Run[CustomExecutor]] = []

    gretel_run_id = 0
    custom_run_id = 0
    for dataset in datasets:
        for source in dataset.sources():
            for model_factory in models:
                model = model_factory()
                if isinstance(model, GretelModel):
                    project_name = f"{project_name_prefix}-{gretel_run_id}"
                    gretel_run_id = gretel_run_id + 1
                    executor = _create_gretel_executor(
                        model=model,
                        project_name=project_name,
                        gretel_sdk=gretel_sdk,
                        gretel_trainer_factory=gretel_trainer_factory,
                        evaluator=evaluator,
                        benchmark_dir=runtime_config.local_dir,
                    )
                    gretel_model_runs.append(
                        Run(
                            identifier=f"gretel-{gretel_run_id}",
                            source=source,
                            executor=executor,
                            status=NotStarted(),
                        )
                    )
                else:
                    custom_run_id = custom_run_id + 1
                    custom_model_runs.append(
                        Run(
                            identifier=f"custom-{custom_run_id}",
                            source=source,
                            executor=CustomExecutor(model=model, evaluator=evaluator),
                            status=NotStarted(),
                        )
                    )

    return Comparison(
        gretel_model_runs=gretel_model_runs,
        custom_model_runs=custom_model_runs,
        runtime_config=runtime_config,
    ).execute()


def _create_gretel_executor(
    model: GretelModel,
    project_name: str,
    gretel_sdk: GretelSDK,
    gretel_trainer_factory: Callable[..., Trainer],
    evaluator: Evaluator,
    benchmark_dir: str,
) -> GretelExecutor:
    if model.config == "AUTO":
        return GretelTrainerExecutor(
            model=model,
            project_name=project_name,
            trainer_factory=gretel_trainer_factory,
            delete_project=gretel_sdk.delete_project,
            benchmark_dir=benchmark_dir,
        )

    try:
        config_dict = read_model_config(model.config)
        model_key = list(config_dict["models"][0])[0]
    except (ModelConfigError, KeyError):
        raise BenchmarkException(f"Invalid Gretel model config for {model.name}")

    if model_key in ("ctgan", "lstm", "synthetics"):
        return GretelTrainerExecutor(
            model=model,
            project_name=project_name,
            trainer_factory=gretel_trainer_factory,
            delete_project=gretel_sdk.delete_project,
            benchmark_dir=benchmark_dir,
        )

    return GretelSDKExecutor(
        project_name=project_name,
        model=model,
        model_key=model_key,
        sdk=gretel_sdk,
        evaluator=evaluator,
    )
