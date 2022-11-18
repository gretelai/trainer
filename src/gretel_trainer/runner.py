"""
Execute a partition strategy using multiple Gretel jobs.

This module will consume a dataset and create a partitioning strategy. This strategy will be saved
to a JSON file. Within each partition there is a ``ctx`` object that will contain metadta about the
various models and handlers used. The context object has the following shape:

{
    "artifact": "artifact used to train model",
    "status": "one of Gretel's job statuses",
    "model_id": "the model ID",
    "attempt": "number of times the model has attempted training",
    "sqs": "an object that contains SQS information from Gretel's API",
    "last_handler": {
        "artifact": "any seed data that was used for generation",
        "attempt": "number of tries to generate",
        "status": "a Gretel job status",
        "handler_id": "the record handler ID",
        "num_records": "how many records were generated, automatically overloaded if seeds are provided"
    }
}
"""
from __future__ import annotations

import json
import logging
import math
import tempfile
import time
from collections import Counter
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from random import choice
from typing import List, Optional, Union

import pandas as pd
import smart_open
from gretel_client.projects import Project
from gretel_client.projects.jobs import ACTIVE_STATES
from gretel_client.projects.models import Model, Status
from gretel_client.projects.records import RecordHandler
from gretel_client.rest import ApiException
from gretel_client.users.users import get_me

from gretel_trainer.strategy import Partition, PartitionConstraints, PartitionStrategy

MODEL_ID = "model_id"
HANDLER_ID = "handler_id"
STATUS = "status"
ARTIFACT = "artifact"
SQS = "sqs"
ATTEMPT = "attempt"
HANDLER = "last_handler"
NUM_RECS = "num_records"

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


@dataclass
class ArtifactResult:
    id: str
    record_count: int


@dataclass
class GenPayload:
    num_records: int
    seed_df: Optional[pd.DataFrame] = None
    seed_artifact_id: Optional[str] = None
    max_invalid: Optional[int] = None


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
    handler_uid: str
    project: Project
    artifact_type: str
    df: pd.DataFrame = None


def _remote_dataframe_fetcher(payload: RemoteDFPayload) -> RemoteDFPayload:
    # We need the model object no matter what
    model = Model(payload.project, model_id=payload.uid)
    job = model

    # if we are downloading handler data, we reset our job
    # to the specific handler object
    if payload.job_type == "run":
        job = RecordHandler(model, record_id=payload.handler_uid)

    download_url = job.get_artifact_link(payload.artifact_type)
    payload.df = pd.read_csv(download_url, compression="gzip")
    return payload


def _maybe_submit_job(
    job: Union[Model, RecordHandler]
) -> Optional[Union[Model, RecordHandler]]:
    try:
        job = job.submit_cloud()
    except ApiException as err:
        if "Maximum number of" in str(err):
            logger.warning(
                "Rate limiting: Max jobs created, skipping new job for now..."
            )
            return None
        else:
            raise

    return job


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
    _handler_status_counter: Counter
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
    def from_completed(
        cls, project: Project, cache_file: Union[str, Path]
    ) -> StrategyRunner:
        cache_file = Path(cache_file)
        if not cache_file.exists():
            raise ValueError("cache file does not exist")

        inst = cls(
            strategy_id="__none__",
            df=None,
            cache_file=cache_file,
            model_config=None,
            partition_constraints=None,
            project=project,
        )
        inst.load()
        return inst

    def _update_job_status(self):
        # Get all jobs that have been created, we can do this
        # by just searching for any partitions have have a "model_id"
        # set
        partitions = self._strategy.query_glob(MODEL_ID, "*")
        # logger.info(f"Fetching updates for {len(partitions)} models...")
        self._status_counter = Counter()
        for partition in partitions:
            last_status = partition.ctx.get(STATUS)

            # If we are done successfully, go on to the next one
            if last_status in (Status.COMPLETED,):
                self._status_counter.update([Status(last_status)])
                continue

            model_id = partition.ctx.get(MODEL_ID)

            # Hydrate a Model object from the remote API
            current_model = Model(self._project, model_id=model_id)

            # Update our strategy-wide counter of model states
            self._status_counter.update([current_model.status])

            # Did our model status change?
            if last_status != current_model.status:
                logger.info(
                    f"Partition {partition.idx} status change from {last_status} to {current_model.status}"
                )

            _update = {STATUS: current_model.status}

            if current_model.status == Status.COMPLETED:
                report = current_model.peek_report()

                if report is None:
                    with smart_open.open(
                        current_model.get_artifact_link("report_json")
                    ) as fin:
                        report = json.loads(fin.read())

                sqs = report["synthetic_data_quality_score"]["score"]
                label = "Moderate"
                if sqs >= 80:
                    label = "Excellent"
                elif sqs >= 60:
                    label = "Good"

                if last_status != current_model.status:
                    logger.info(
                        f"Partition {partition.idx} completes with SQS: {label} ({sqs})"
                    )

                _update.update({SQS: report})

            partition.update_ctx(_update)
            self._strategy.status_counter = dict(self._status_counter)
            # Aggressive, but save after every update
            self._strategy.status_counter = dict(self._status_counter)
            self._strategy.save()

        # If every partition is done, we may not have saved the strategy
        self._strategy.status_counter = dict(self._status_counter)
        self._strategy.save()

    def _update_handler_status(self):
        partitions = self._strategy.partitions
        self._handler_status_counter = Counter()
        for partition in partitions:
            model_id = partition.ctx.get(MODEL_ID)
            handler_id = partition.ctx.get(HANDLER, {}).get(HANDLER_ID)
            last_status = partition.ctx.get(HANDLER, {}).get(STATUS)

            # No need to refresh completed handlers from the remote API
            if last_status in (Status.COMPLETED,):
                self._handler_status_counter.update([Status(last_status)])
                continue

            if not handler_id:
                continue

            # Hydrate a Model and Handler object from the remote API
            current_model = Model(self._project, model_id=model_id)
            current_handler = RecordHandler(model=current_model, record_id=handler_id)

            self._handler_status_counter.update([current_handler.status])

            if last_status != current_handler.status:
                logger.info(
                    f"Partition {partition.idx} record generation status change from {last_status} to {current_handler.status}"
                )

            partition.ctx[HANDLER][STATUS] = current_handler.status
            self._strategy.save()

    @_needs_load
    def cancel_all(self):
        partitions = self._strategy.query_glob(MODEL_ID, "*")
        for partition in partitions:
            model_id = partition.ctx.get(MODEL_ID)

            # Hydrate a Model object from the remote API
            current_model = Model(self._project, model_id=model_id)
            logger.warning(f"Cancelling: {current_model.id}")
            current_model.cancel()

    def _refresh_max_job_capacity(self):
        self._max_jobs_active = get_me()["service_limits"]["max_jobs_active"]

    @property
    @_needs_load
    def has_capacity(self) -> bool:
        num_active = len(self._gather_statuses(ACTIVE_STATES))
        self._refresh_max_job_capacity()
        return num_active < self._max_jobs_active

    def _remove_unused_artifact(self) -> Optional[str]:
        project_artifacts = self._project.artifacts
        curr_artifacts = set()

        if len(project_artifacts) < self._max_artifacts:
            return "__none__"

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
                logger.debug(f"Attempting to remove artifact: {p.ctx.get(ARTIFACT)}")
                self._project.delete_artifact(artifact_key)
                p.update_ctx({ARTIFACT: None})
                self._strategy.save()
                return artifact_key

            # try and remove a seed artifact if one exists
            handler_status = p.ctx.get(HANDLER, {}).get(STATUS)
            handler_artifact_key = p.ctx.get(HANDLER, {}).get(ARTIFACT)

            if handler_status is None or handler_status in ACTIVE_STATES:
                continue

            if handler_artifact_key:
                logger.debug(
                    f"Attempting to remove handler artifact: {handler_artifact_key}"
                )
                self._project.delete_artifact(handler_artifact_key)
                p.ctx[HANDLER][ARTIFACT] = None
                return handler_artifact_key

        # If we couldn't remove an artifact from this current strategy,
        # we'll just remove some other random one
        try:
            logger.debug("Removing artifact not belonging to this Strategy...")
            for art in project_artifacts:
                key = art.get("key")
                if key in curr_artifacts:
                    continue
                self._project.delete_artifact(key)
                return key
        except Exception as err:
            logger.warning(f"Could not delete artifact: {str(err)}")

        return None

    def _partition_to_artifact(self, partition: Partition) -> Optional[ArtifactResult]:
        removed_artifact = self._remove_unused_artifact()
        if not removed_artifact:
            logger.debug("Could not make room for next data set, waiting for room...")
            # We couldn't make room so we don't try and upload the next artifact
            return None

        filename = f"{self.strategy_id}-{partition.idx}.csv"
        df_to_upload = partition.extract_df(self._df)
        res = self._df_to_artifact(df_to_upload, filename)
        partition.update_ctx({ARTIFACT: res.id})
        self._strategy.save()
        return res

    def _df_to_artifact(self, df: pd.DataFrame, filename: str) -> ArtifactResult:
        with tempfile.TemporaryDirectory() as tmp:
            target_file = str(Path(tmp) / filename)
            df.to_csv(target_file, index=False)
            artifact_id = self._project.upload_artifact(target_file)
            return ArtifactResult(id=artifact_id, record_count=len(df))

    @_needs_load
    def train_partition(
        self, partition: Partition, artifact: ArtifactResult
    ) -> Optional[str]:
        attempt = partition.ctx.get(ATTEMPT, 0) + 1
        model_config = deepcopy(self._model_config)
        data_source = None

        if "synthetics" in model_config["models"][0].keys():

            # If we're trying this model for a second+ time, we reduce the vocab size to
            # utilize the char encoder in order to give a better chance and success
            if attempt > 1:
                learning_rate = choice([0.001, 0.01])
                vocab_size = choice([20000, 0])
                vocab_description = "enabled" if vocab_size > 0 else "disabled"

                logger.info(
                    f"Modifying configuration parameters. Setting learning rate to {learning_rate}. SentencePiece tokenizer {vocab_description}."
                )
                model_config["models"][0]["synthetics"]["params"][
                    "learning_rate"
                ] = learning_rate
                model_config["models"][0]["synthetics"]["params"][
                    "vocab_size"
                ] = vocab_size

            # If this partition is for the first-N headers and we have known seed headers, we have to
            # modify the configuration to account for the seed task.
            if partition.columns.seed_headers:
                model_config["models"][0]["synthetics"]["task"] = {
                    "type": "seed",
                    "attrs": {"fields": partition.columns.seed_headers},
                }

        model = self._project.create_model_obj(
            data_source=artifact.id, model_config=model_config,
        )
        model.name = artifact.id.split("_")[-1]

        model = _maybe_submit_job(model)
        if model is None:
            return None

        partition.ctx.update(
            {
                STATUS: model.status,
                ARTIFACT: artifact.id,
                MODEL_ID: model.model_id,
                ATTEMPT: attempt,
            }
        )
        self._strategy.save()
        logger.info(
            f"Started model: {model.print_obj['model_name']} " f"source: {artifact.id}"
        )
        return model.model_id

    @_needs_load
    def run_partition(
        self, partition: Partition, gen_payload: GenPayload
    ) -> Optional[str]:
        """
        Run a record handler for a model and return the job id.

        NOTE: This assumes the partition is successfully trained and has an
        available model.
        """
        handler_dict = partition.ctx.get(HANDLER)
        if handler_dict is None:
            partition.ctx[HANDLER] = {}
        attempt = partition.ctx.get(HANDLER).get(ATTEMPT, 0) + 1
        model_id = partition.ctx.get(MODEL_ID)

        # Hydrate our trained model so we can start the handler
        model_obj = Model(self._project, model_id=model_id)

        # Create and start our handler to generate data
        handler_obj = model_obj.create_record_handler_obj(
            data_source=gen_payload.seed_artifact_id,
            params={
                "num_records": gen_payload.num_records,
                "max_invalid": gen_payload.max_invalid,
            },
        )
        handler_obj = _maybe_submit_job(handler_obj)
        if handler_obj is None:
            return None

        _ctx_update = {
            ATTEMPT: attempt,
            ARTIFACT: gen_payload.seed_artifact_id,
            NUM_RECS: gen_payload.num_records,
            STATUS: handler_obj.status,
            HANDLER_ID: handler_obj.record_id,
        }

        partition.ctx[HANDLER].update(_ctx_update)
        self._strategy.save()
        logger.info(
            f"Generating {gen_payload.num_records} records from model: {model_obj.print_obj['model_name']}"
        )
        return handler_obj.record_id

    @_needs_load
    def train_next_partition(self) -> Optional[str]:
        start_job = False
        for partition in self._strategy.partitions:
            status = partition.ctx.get(STATUS)  # type: Status

            # If we've never done a job for this partition, we should start one
            if status is None:
                logger.info(
                    f"Partition {partition.idx} is new, starting model creation"
                )
                start_job = True

            # If the job failed, should we try again?
            elif (
                status in (Status.ERROR, Status.LOST)
                and partition.ctx.get(ATTEMPT, 0) < self._error_retry_limit
            ):
                logger.info(
                    f"Partition {partition.idx} status: {status.value}, re-attempting job"
                )
                start_job = True

            if start_job:
                artifact = self._partition_to_artifact(partition)
                if artifact.id is None:
                    return None
                return self.train_partition(partition, artifact)

    @_needs_load
    def run_next_partition(self, gen_payload: GenPayload) -> Optional[str]:
        start_job = False
        for partition in self._strategy.partitions:
            status = partition.ctx.get(HANDLER, {}).get(STATUS)  # type: Status
            attempt_count = partition.ctx.get(HANDLER, {}).get(ATTEMPT, 0)

            if status is None:
                logger.info(f"Generating data for partition {partition.idx}")
                start_job = True

            elif (
                status in (Status.ERROR, Status.LOST)
                and attempt_count < self._error_retry_limit
            ):
                logger.info(
                    f"Partition {partition.idx} has status {status.value}, re-attempting generation"
                )
                start_job = True

            if start_job:
                use_seeds = False
                # If this partition has seed fields and we were given seeds, we need to upload
                # the artifact first.
                if partition.columns.seed_headers and isinstance(
                    gen_payload.seed_df, pd.DataFrame
                ):
                    # NOTE(jm): If we've tried N-1 attempts with seeds and the handler has continued
                    # fail then we should not use seeds to at least let the handler try to succeed.
                    # One example of this happening would be when a partition's model receives seeds
                    # where the values of the seeds were not in the training set (due to partitioning).
                    if attempt_count == self._error_retry_limit - 1:
                        logger.info(
                            f"WARNING: Disabling seeds for partition {partition.idx} due to previous failed generation attempts..."
                        )
                    else:
                        logger.info(
                            "Partition has seed fields, uploading seed artifact..."
                        )
                        use_seeds = True
                        removed_artifact = self._remove_unused_artifact()
                        if removed_artifact is None:
                            logger.info(
                                "Could not start generation with seeds, an old artifact could not be removed"
                            )
                            return None

                        filename = f"{self.strategy_id}-seeds-{partition.idx}.csv"
                        artifact = self._df_to_artifact(gen_payload.seed_df, filename)

                new_payload = GenPayload(
                    num_records=gen_payload.num_records,
                    max_invalid=gen_payload.max_invalid,
                    seed_artifact_id=artifact.id if use_seeds else None,
                )

                return self.run_partition(partition, new_payload)

    @_needs_load
    def clear_partition_runs(self):
        """
        Partitions should only be trained until they are 'completed', however we can run
        a partition any number of times. Before we do that, we want to go through and
        """
        for partition in self._strategy.partitions:
            partition.ctx[HANDLER] = {}

    def _gather_statuses(self, statuses: List[Status]) -> List[Partition]:
        out = []
        for partition in self._strategy.partitions:
            status = partition.ctx.get(STATUS)
            if status is None:
                continue
            if status in statuses:
                out.append(partition)
        return out

    @_needs_load
    def is_done(self, *, handler: bool = False) -> bool:
        done = 0
        for p in self._strategy.partitions:
            if handler:
                ctx_base = p.ctx.get(HANDLER, {})
            else:
                ctx_base = p.ctx

            status = ctx_base.get(STATUS)
            attempt = ctx_base.get(ATTEMPT, 0)
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
        logger.info(f"Processing {len(self._strategy.partitions)} partitions")
        while True:
            self._update_job_status()
            if not self.has_capacity:
                logger.debug("At active capacity, waiting for more...")
                time.sleep(10)
                continue

            model_id = self.train_next_partition()
            if model_id:
                continue

            if self.is_done():
                break

            time.sleep(10)

        logger.info(dict(self._status_counter))

    @_needs_load
    def _get_synthetic_data(self, job_type: str, artifact_type: str) -> pd.DataFrame:
        if job_type == "model":
            self._update_job_status()
            num_completed = self._status_counter.get(Status.COMPLETED, 0)
        elif job_type == "run":
            self._update_handler_status()
            num_completed = self._handler_status_counter.get(Status.COMPLETED, 0)
        else:
            raise ValueError("invalid job_type")

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
            # NOTE: When sending the remote DF payload we try to extract both
            # model and handler IDs and the thread workers can interpret which
            # ones they need to use.
            payload = RemoteDFPayload(
                partition=partition.idx,
                slot=partition.columns.idx,
                job_type=job_type,
                handler_uid=partition.ctx.get(HANDLER, {}).get(HANDLER_ID),
                uid=partition.ctx.get(MODEL_ID),
                project=self._project,
                artifact_type=artifact_type,
            )
            futures.append(pool.submit(_remote_dataframe_fetcher, payload))

        wait(futures, return_when=ALL_COMPLETED)

        for future in futures:
            payload = future.result()  # type: RemoteDFPayload

            curr_df = df_chunks[payload.slot]
            df_chunks[payload.slot] = pd.concat([curr_df, payload.df]).reset_index(
                drop=True
            )

        df = pd.concat(list(df_chunks.values()), axis=1)
        return df

    def _maybe_restore_df_headers(self, df) -> pd.DataFrame:
        if isinstance(self._df, pd.DataFrame):
            return df[self._df.columns]

        if self._strategy.original_headers:
            return df[self._strategy.original_headers]

        return df

    def get_training_synthetic_data(self) -> pd.DataFrame:
        df = self._get_synthetic_data("model", "data_preview")
        return self._maybe_restore_df_headers(df)

    def get_synthetic_data(self) -> pd.DataFrame:
        df = self._get_synthetic_data("run", "data")
        return self._maybe_restore_df_headers(df)

    def get_sqs_information(self) -> List[dict]:
        return [
            partition.ctx[SQS]
            for partition in self._strategy.partitions
        ]

    @_needs_load
    def generate_data(
        self,
        *,
        seed_df: Optional[pd.DataFrame] = None,
        num_records: Optional[int] = None,
        max_invalid: Optional[int] = None,
        clear_cache: bool = False,
    ):

        if seed_df is None and not num_records:
            raise ValueError("must provide a seed_df or num_records to generate")

        if isinstance(seed_df, pd.DataFrame) and num_records:
            raise ValueError("must use one of seed_df or num_records only")

        # Refresh all of the trained models
        logger.info("Loading existing model information...")
        self._update_job_status()

        # We can't generate a dataset if any of the models are in a bad state, so we check that here
        completed_count = self._status_counter[Status.COMPLETED]
        if completed_count != self._strategy.partition_count:
            raise RuntimeError(
                f"Cannot generate data, {self._strategy.partition_count - completed_count} partitions do not have a completed model."
            )

        # Clear out all previous record handler metadata
        if clear_cache:
            self.clear_partition_runs()

        # If we have seeds, then we use the number of seeds as the number of records
        # to generate from each model.
        found_seeds = False
        if isinstance(seed_df, pd.DataFrame):
            num_records = len(seed_df)

            # Loop through all of the partitions and make sure we have some that
            # take seed values, if we don't have any partitions set for seeds
            # and we recieved a seed DF, we should error.
            for partition in self._strategy.partitions:
                if partition.columns.seed_headers:
                    found_seeds = True
                    break

            if not found_seeds:
                raise RuntimeError(
                    "You cannot provide a seed_df since models were not conditioned with seed headers."
                )

        # NOTE: This payload will be used to create a new payload object per
        # partition, so this will get passed in more as a template to the next
        # routine
        partition_num_records = math.ceil(num_records / self._strategy.row_partition_count)
        gen_payload = GenPayload(
            seed_df=seed_df, num_records=partition_num_records, max_invalid=max_invalid
        )

        logger.info(
            f"Starting data generation from {self._strategy.partition_count} models"
        )

        update_remote = True

        while True:
            # Try and start as many jobs as we can, until we hit a rate
            # limit error, then we can check our capacity availabiliy and
            # wait until trying more.
            #
            # The main idea is that we want to avoid polling for re-hydration
            # so much since that can take so long.
            if update_remote:
                logger.info("Refreshing generation statuses...")
                self._update_handler_status()
                update_remote = False

            try:
                handler_id = self.run_next_partition(gen_payload)
                if handler_id is not None:
                    # Go around and try again if this succeeded
                    continue

            # Catch a 4xx in the event we are at capacity, or something else goes wrong
            except Exception as err:
                logger.warning(f"Error running model: {str(err)}")

            # Are we done done?
            if self.is_done(handler=True):
                break

            # Wait until we have capacity, once we have capacity, we'll
            # set our flag to refresh the remote status update
            while True:
                if not self.has_capacity:
                    logger.info("Waiting for more job capacity.")
                    time.sleep(10)
                else:
                    update_remote = True
                    break

            time.sleep(10)

        logger.info(dict(self._handler_status_counter))
