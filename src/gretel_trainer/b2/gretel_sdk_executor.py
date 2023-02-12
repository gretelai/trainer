import time
import logging
from multiprocessing.managers import DictProxy
from typing import Callable, Optional, Tuple

import pandas as pd
from gretel_client.projects.jobs import END_STATES, Job, Status
from gretel_client.projects.models import Model, read_model_config
from gretel_client.projects.projects import Project
from gretel_client.projects.records import RecordHandler

from gretel_trainer.b2.core import BenchmarkException, Dataset, RunIdentifier, Timer
from gretel_trainer.b2.gretel_models import GretelModel, GretelModelConfig
from gretel_trainer.b2.status import NotStarted, InProgress, Completed, Failed, RunStatus

logger = logging.getLogger(__name__)


class GretelSDKExecutor:
    def __init__(
        self,
        project: Project,
        benchmark_model: GretelModel,
        dataset: Dataset,
        run_identifier: RunIdentifier,
        statuses: DictProxy,
        refresh_interval: int,
    ):
        self.project = project
        self.benchmark_model = benchmark_model
        self.dataset = dataset
        self.refresh_interval = refresh_interval
        self.run_identifier = run_identifier
        self.statuses = statuses
        self.set_status(NotStarted())

        self.model: Optional[Model] = None
        self.record_handler: Optional[RecordHandler] = None
        self.train_time: Optional[float] = None
        self.generate_time: Optional[float] = None

    def set_status(self, status: RunStatus) -> None:
        self.status = status
        self.statuses[self.run_identifier] = status

    def _format_model_config(self) -> GretelModelConfig:
        config = read_model_config(self.benchmark_model.config)
        config["name"] = f"{self.run_identifier[0]}-{self.run_identifier[1]}"
        return config

    def train(self) -> None:
        # TODO: potentially skip (datatype, col count, etc.)
        logger.info(f"Starting model training for run `{self.run_identifier}`")
        self.set_status(InProgress(stage="train"))
        model_config = self._format_model_config()
        self.model = self.project.create_model_obj(
            model_config=model_config, data_source=self.dataset.data_source
        )
        train_time = Timer()
        with train_time:
            self.model.submit_cloud()
            job_status = _await_job(self.model, self.refresh_interval)
        self.train_time = train_time.duration()
        if job_status in END_STATES and job_status != Status.COMPLETED:
            self.set_status(Failed(during="train", train_secs=self.train_time))
            logger.info(f"Run `{self.run_identifier}` failed")

    def generate(self) -> None:
        if self.model is None or self.train_time is None:
            raise BenchmarkException("Cannot generate before training")

        if isinstance(self.status, Failed):
            return None

        logger.info(f"Starting synthetic data generation for run `{self.run_identifier}`")
        self.set_status(InProgress(stage="generate", train_secs=self.train_time))
        self.record_handler = self.model.create_record_handler_obj(
            params={"num_records": self.dataset.row_count}
        )
        generate_time = Timer()
        with generate_time:
            self.record_handler.submit_cloud()
            job_status = _await_job(self.record_handler, self.refresh_interval)
        self.generate_time = generate_time.duration()
        if job_status == Status.COMPLETED:
            self.set_status(Completed(
                sqs=self.get_sqs_score(),
                train_secs=self.train_time,
                generate_secs=self.generate_time,
                synthetic_data=self.get_synthetic_data(),
            ))
            logger.info(f"Run `{self.run_identifier}` completed successfully")
        elif job_status in END_STATES:
            self.set_status(Failed(
                during="generate",
                train_secs=self.train_time,
                generate_secs=self.generate_time,
            ))
            logger.info(f"Run `{self.run_identifier}` failed")

    def get_sqs_score(self) -> int:
        if self.model is None:
            raise BenchmarkException("Cannot get SQS score before training")

        sqs_score = 0
        summary = self.model.get_report_summary()
        for stat in summary["summary"]:
            if stat["field"] == "synthetic_data_quality_score":
                sqs_score = stat["value"]

        return sqs_score

    def get_synthetic_data(self) -> pd.DataFrame:
        if self.record_handler is None:
            raise BenchmarkException("Cannot get synthetic data before generating")

        return pd.read_csv(self.record_handler.get_artifact_link("data"), compression="gzip")


def _await_job(job: Job, refresh_interval: int) -> Status:
    failed_refresh_attempts = 0
    status = job.status
    while not _finished(status):
        if failed_refresh_attempts >= 5:
            return Status.LOST
        time.sleep(refresh_interval)
        status, failed_refresh_attempts = _cautiously_refresh_status(job, failed_refresh_attempts)
    return status


def _finished(status: Status) -> bool:
    return status in END_STATES


def _cautiously_refresh_status(job: Job, attempts: int) -> Tuple[Status, int]:
    try:
        job.refresh()
        attempts = 0
    except:
        attempts = attempts + 1

    return (job.status, attempts)
