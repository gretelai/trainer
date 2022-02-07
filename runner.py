from __future__ import annotations

import tempfile
import time

from collections import Counter
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import List, Optional, Union, Tuple, NamedTuple

import pandas as pd

from strategy import Partition, PartitionConstraints, PartitionStrategy

from gretel_client.projects import Project
from gretel_client.projects.jobs import ACTIVE_STATES
from gretel_client.projects.models import Model, Status
from gretel_client.users.users import get_me

MODEL_ID = "model_id"
STATUS = "status"
ARTIFACT = "artifact"
SQS = "sqs"
ATTEMPT = "attempt"


class Artifact(NamedTuple):
    id: str
    record_count: int


def _needs_load(func):
    @wraps(func)
    def wrapper(inst: StrategyRunner, *args, **kwargs):
        if not inst._loaded:
            inst.load()
        return func(inst, *args, **kwargs)

    return wrapper


@dataclass
class RemoteDFPayload:
    partition: int
    slot: int
    job_type: str
    uid: str
    project: Project
    artifact_type: str
    df: pd.DataFrame = None


def _remote_dataframe_fetcher(payload: RemoteDFPayload) -> RemoteDFPayload:
    if payload.job_type == "model":
        job = Model(payload.project, model_id=payload.uid)

    download_url = job.get_artifact_link(payload.artifact_type)
    payload.df = pd.read_csv(download_url, compression="gzip")
    return payload


class StrategyRunner:

    _df: pd.DataFrame
    _cache_file: Path
    _constraints = PartitionConstraints
    _strategy = PartitionStrategy
    _model_config: dict
    _max_jobs_active: int
    _project: Project
    _loaded: bool
    _artifacts: List[str]
    _cache_overwrite: bool
    _max_artifacts: int = 25
    _status_counter: Counter
    _error_retry_limit: int
    strategy_id: str

    def __init__(
        self,
        *,
        strategy_id: str,
        df: pd.DataFrame,
        cache_file: Union[str, Path],
        cache_overwrite: bool = False,
        model_config: dict,
        partition_constraints: PartitionConstraints,
        project: Project,
        error_retry_limit: int = 3,
    ):
        self._df = df
        self._cache_file = Path(cache_file)
        self._constraints = partition_constraints
        self._model_config = model_config
        self._project = project
        self._loaded = False
        self._cache_overwrite = cache_overwrite
        self._artifacts = []
        self.strategy_id = strategy_id
        self._status_counter = Counter()
        self._error_retry_limit = error_retry_limit

    def load(self):
        """Hydrate the instance before we can start
        doing work, must be called after init
        """
        if self._loaded:
            return

        self._refresh_max_job_capacity()

        # If the cache file exists, we'll try and load an existing
        # strategy. If not, we'll create a new strategy with the
        # provided constraints.

        if self._cache_file.exists() and not self._cache_overwrite:
            self._strategy = PartitionStrategy.from_disk(self._cache_file)
        else:
            self._strategy = PartitionStrategy.from_dataframe(
                self.strategy_id, self._df, self._constraints
            )

        self._strategy.save_to(self._cache_file, overwrite=True)

        self._loaded = True

    @classmethod
    def from_completed(cls, project: Project, cache_file: Union[str, Path]):
        cache_file = Path(cache_file)
        if not cache_file.exists():
            raise ValueError("cache file does not exist")

        return cls(
            strategy_id="__none__",
            df=None,
            cache_file=cache_file,
            model_config=None,
            partition_constraints=PartitionConstraints(max_row_partitions=1),
            project=project,
        )

    def _update_job_status(self):
        # Get all jobs that have been created, we can do this
        # by just searching for any partitions have have a "model_id"
        # set
        partitions = self._strategy.query_glob(MODEL_ID, "*")
        # print(f"Fetching updates for {len(partitions)} models...")
        self._status_counter = Counter()
        for partition in partitions:
            model_id = partition.ctx.get(MODEL_ID)

            # Hydrate a Model object from the remote API
            current_model = Model(self._project, model_id=model_id)

            # Update our strategy-wide counter of model states
            self._status_counter.update([current_model.status])

            last_status = partition.ctx.get(STATUS)

            # Did our model status change?
            if last_status != current_model.status:
                print(
                    f"Partition {partition.idx} status change from {last_status} to {current_model.status}"
                )

            _update = {STATUS: current_model.status}

            if current_model.status == Status.COMPLETED:
                report = current_model.peek_report()
                sqs = report['synthetic_data_quality_score']['score']
                label = "Moderate"
                if sqs >= 80:
                    label = "Excellent"
                elif sqs >= 60:
                    label = "Good"

                if last_status != current_model.status:
                    print(f"Partition {partition.idx} completes with SQS: {label} ({sqs})")

                _update.update({SQS: report})
            partition.update_ctx(_update)

            # Aggressive, but save after every update
            self._strategy.save()

    @_needs_load
    def cancel_all(self):
        partitions = self._strategy.query_glob(MODEL_ID, "*")
        for partition in partitions:
            model_id = partition.ctx.get(MODEL_ID)

            # Hydrate a Model object from the remote API
            current_model = Model(self._project, model_id=model_id)
            print(f"Cancelling: {current_model.id}")
            current_model.cancel()

    def _refresh_max_job_capacity(self):
        self._max_jobs_active = get_me()["service_limits"]["max_jobs_active"]

    @property
    @_needs_load
    def has_capacity(self) -> bool:
        num_active = len(self._gather_statuses(ACTIVE_STATES))
        self._refresh_max_job_capacity()
        return num_active < self._max_jobs_active

    def _partition_to_artifact(self, partition: Partition) -> Artifact:
        project_artifacts = self._project.artifacts
        curr_artifacts = set()

        if len(project_artifacts) >= self._max_artifacts:
            found_artifact = False

            # First we try and remove an artifact from this strategy by
            # looking at our model states
            for p in self._strategy.partitions:
                status = p.ctx.get(STATUS)
                artifact_key = p.ctx.get(ARTIFACT)

                curr_artifacts.add(artifact_key)

                # We don't want to delete an artifact that is maybe
                # about to be used
                if status is None or status in ACTIVE_STATES:
                    continue

                if artifact_key:
                    print(f"Attempting to remove artifact: {p.ctx.get(ARTIFACT)}")
                    self._project.delete_artifact(artifact_key)
                    p.update_ctx({ARTIFACT: None})
                    self._strategy.save()
                    found_artifact = True
                    break

            # If we couldn't remove an artifact from this current strategy,
            # we'll just remove some other random one
            if not found_artifact:
                try:
                    print("Removing artifact not belonging to this Strategy...")
                    for art in project_artifacts:
                        key = art.get("key")
                        if key in curr_artifacts:
                            continue
                        self._project.delete_artifact(key)
                        found_artifact = True
                        break
                except Exception as err:
                    print(f"Could not delete artifact: {str(err)}")

                if not found_artifact:
                    print("Could not make room for next dataset, waiting for room...")
                    # We couldn't make room so we don't try and upload the next artifact
                    return None

        filename = f"{self.strategy_id}-{partition.idx}.csv"
        df_to_upload = partition.extract_df(self._df)
        with tempfile.TemporaryDirectory() as tmp:
            target_file = str(Path(tmp) / filename)
            df_to_upload.to_csv(target_file, index=False)
            artifact_id = self._project.upload_artifact(target_file)
        partition.update_ctx({ARTIFACT: artifact_id})
        self._strategy.save()
        return Artifact(artifact_id, len(df_to_upload))

    @_needs_load
    def train_partition(self, partition: Partition, artifact: Artifact) -> str:

        model_config = deepcopy(self._model_config)
        model_config['models'][0]['synthetics']['generate'] = {
            "num_records": artifact.record_count, "max_invalid": None}

        model = self._project.create_model_obj(
            model_config=model_config, data_source=artifact.id
        )
        model = model.submit_cloud()

        attempt = partition.ctx.get(ATTEMPT, 0) + 1
        partition.ctx.update(
            {
                STATUS: model.status,
                ARTIFACT: artifact.id,
                MODEL_ID: model.model_id,
                ATTEMPT: attempt,
            }
        )
        self._strategy.save()
        print(f"Started model: {model.print_obj['model_name']} "
              f"source: {model.print_obj['config']['models'][0]['synthetics']['data_source']}")
        return model.model_id

    @_needs_load
    def train_next_partition(self) -> Optional[str]:
        start_job = False
        for partition in self._strategy.partitions:
            status = partition.ctx.get(STATUS)  # type: Status

            # If we've never done a job for this partition, we should start one
            if status is None:
                print(f"Partition {partition.idx} is new, starting model creation")
                start_job = True

            # If the job failed, should we try again?
            elif (
                status in (Status.ERROR, Status.LOST)
                and partition.ctx.get(ATTEMPT, 0) < self._error_retry_limit
            ):
                print(
                    f"Partition {partition.idx} status: {status.value}, re-attempting job"
                )
                start_job = True

            if start_job:
                artifact = self._partition_to_artifact(partition)
                if artifact.id is None:
                    return None
                return self.train_partition(partition, artifact)

    def _gather_statuses(self, statuses: List[Status]) -> List[Partition]:
        out = []
        for partition in self._strategy.partitions:
            status = partition.ctx.get(STATUS)
            if status is None:
                continue
            if status in statuses:
                out.append(partition)
        return out

    @property
    @_needs_load
    def is_done_training(self) -> bool:
        done = 0
        for p in self._strategy.partitions:
            status = p.ctx.get(STATUS)
            attempt = p.ctx.get(ATTEMPT, 0)
            if status is None:
                continue

            if status in (Status.COMPLETED, Status.CANCELLED):
                done += 1
            elif (
                status in (Status.ERROR, Status.LOST)
                and attempt >= self._error_retry_limit
            ):
                done += 1

        return done == len(self._strategy.partitions)

    @_needs_load
    def train_all_partitions(self):
        print(f"Processing {len(self._strategy.partitions)} partitions")
        while True:
            self._update_job_status()
            if not self.has_capacity:
                print("At active capacity, waiting for more...")
                time.sleep(10)
                continue

            model_id = self.train_next_partition()
            if model_id:
                continue

            if self.is_done_training:
                break

            time.sleep(10)

        print(dict(self._status_counter))

    @_needs_load
    def get_training_synthetic_data(self) -> pd.DataFrame:
        self._update_job_status()
        num_completed = self._status_counter.get(Status.COMPLETED, 0)
        if num_completed != self._strategy.partition_count:
            raise RuntimeError(
                "Not all partitions are completed, cannot fetch synthetic data from trained models"
            )

        # We will have at least one column-wise DF, this holds
        # one DF for each header cluster we have
        df_chunks = {
            i: pd.DataFrame() for i in range(0, self._strategy.header_cluster_count)
        }

        pool = ThreadPoolExecutor()
        futures = []
        for partition in self._strategy.partitions:
            payload = RemoteDFPayload(
                partition=partition.idx,
                slot=partition.columns.idx,
                job_type="model",
                uid=partition.ctx.get(MODEL_ID),
                project=self._project,
                artifact_type="data_preview",
            )
            futures.append(pool.submit(_remote_dataframe_fetcher, payload))

        wait(futures, return_when=ALL_COMPLETED)

        for future in futures:
            payload = future.result()  # type: RemoteDFPayload

            curr_df = df_chunks[payload.slot]
            df_chunks[payload.slot] = pd.concat([curr_df, payload.df]).reset_index(
                drop=True
            )

        return pd.concat(list(df_chunks.values()), axis=1)
