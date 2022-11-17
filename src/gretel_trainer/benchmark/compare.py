import concurrent.futures as futures
import multiprocessing as mp
import shutil
import time

from contextlib import suppress
from dataclasses import dataclass
from multiprocessing.managers import DictProxy
from typing import Callable, Dict, List, Type, Union

import pandas as pd

from gretel_trainer.benchmark.core import (
    BenchmarkException,
    Completed,
    Dataset,
    execute,
    Failed,
    InProgress,
    ModelFactory,
    NotStarted,
    Run,
    Skipped,
)
from gretel_trainer.benchmark.custom.executor import CustomExecutor
from gretel_trainer.benchmark.gretel.executor import GretelExecutor
from gretel_trainer.benchmark.gretel.models import GretelModel
from gretel_trainer.benchmark.gretel.sdk import GretelSDK
from gretel_trainer.benchmark.gretel.trainer import TrainerFactory

from gretel_client.projects.models import read_model_config


@dataclass
class RuntimeConfig:
    local_dir: str
    project_prefix: str
    thread_pool: futures.Executor
    wait_secs: float
    auto_clean: bool


class Comparison:
    def __init__(
        self,
        gretel_model_runs: List[Run[GretelExecutor]],
        custom_model_runs: List[Run[CustomExecutor]],
        runtime_config: RuntimeConfig,
        gretel_sdk: GretelSDK,
    ):
        self.gretel_model_runs = gretel_model_runs
        self.custom_model_runs = custom_model_runs
        self.runtime_config = runtime_config
        self.gretel_sdk = gretel_sdk
        self._manager = mp.Manager()
        self.results_dict = self._manager.dict()
        for run in self._all_runs:
            self.results_dict[run.identifier] = NotStarted()
        self.futures = []

    @property
    def _all_runs(self) -> List[Run]:
        return self.gretel_model_runs + self.custom_model_runs

    @property
    def is_complete(self) -> bool:
        return _is_complete(self.results_dict)

    @property
    def results(self) -> pd.DataFrame:
        result_records = [_result_dict(run, self.results_dict) for run in self._all_runs]
        return pd.DataFrame.from_records(result_records)

    def export_results(self, destination: str):
        self.results.to_csv(destination, index=False)

    def execute(self):
        for run in self.gretel_model_runs:
            self.futures.append(
                self.runtime_config.thread_pool.submit(execute, run, self.results_dict)
            )

        for run in self.custom_model_runs:
            execute(run, self.results_dict)

        self.futures.append(
            self.runtime_config.thread_pool.submit(
                _cleanup, self.results_dict, self.runtime_config, self.gretel_sdk
            )
        )

        return self

    def wait(self):
        [future.result() for future in self.futures]
        return self


def _cleanup(results_dict: DictProxy, runtime_config: RuntimeConfig, sdk: GretelSDK) -> None:
    if runtime_config.auto_clean:
        while not _is_complete(results_dict):
            time.sleep(runtime_config.wait_secs)

        with suppress(Exception):
            for project in sdk.search_projects(runtime_config.project_prefix):
                project.delete()
            shutil.rmtree(runtime_config.local_dir)


def _is_complete(results_dict: DictProxy) -> bool:
    return all(
        isinstance(status, (Completed, Failed, Skipped))
        for status in results_dict.values()
    )


def _result_dict(run: Run, results_dict: DictProxy) -> Dict:
    status = results_dict[run.identifier]

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
        "Input data": run.source.name,
        "Model": run.executor.model_name,
        "DataType": run.source.datatype,
        "Rows": run.source.row_count,
        "Columns": run.source.column_count,
        "Status": status.display,
        "SQS": sqs,
        "Train time (sec)": train_time,
        "Generate time (sec)": generate_time,
        "Total time (sec)": total_time,
    }


def compare(
    *,
    datasets: List[Dataset],
    models: List[Union[ModelFactory, Type[GretelModel]]],
    runtime_config: RuntimeConfig,
    gretel_sdk: GretelSDK,
    gretel_trainer_factory: TrainerFactory,
) -> Comparison:
    gretel_sdk.configure_session()

    gretel_model_runs: List[Run[GretelExecutor]] = []
    custom_model_runs: List[Run[CustomExecutor]] = []

    gretel_run_id = 0
    custom_run_id = 0
    for dataset in datasets:
        for source in dataset.sources():
            for model_factory in models:
                model = model_factory()
                if isinstance(model, GretelModel):
                    project_name = f"{runtime_config.project_prefix}-{gretel_run_id}"
                    run_identifier = f"gretel-{gretel_run_id}"
                    gretel_run_id = gretel_run_id + 1
                    executor = GretelExecutor(
                        model=model,
                        project_name=project_name,
                        sdk=gretel_sdk,
                        trainer_factory=gretel_trainer_factory,
                        benchmark_dir=runtime_config.local_dir,
                    )
                    gretel_model_runs.append(
                        Run(
                            identifier=run_identifier,
                            source=source,
                            executor=executor,
                        )
                    )
                else:
                    run_identifier = f"custom-{custom_run_id}"
                    custom_run_id = custom_run_id + 1
                    executor = CustomExecutor(model=model, evaluate=gretel_sdk.evaluate)
                    custom_model_runs.append(
                        Run(
                            identifier=run_identifier,
                            source=source,
                            executor=executor,
                        )
                    )

    return Comparison(
        gretel_model_runs=gretel_model_runs,
        custom_model_runs=custom_model_runs,
        runtime_config=runtime_config,
        gretel_sdk=gretel_sdk,
    ).execute()
