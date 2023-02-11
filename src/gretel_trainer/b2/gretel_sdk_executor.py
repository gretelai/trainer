import time
import logging
from multiprocessing.managers import DictProxy
from typing import Callable, Optional, Tuple

import pandas as pd
from gretel_client.projects.jobs import END_STATES, Job, Status
from gretel_client.projects.models import Model
from gretel_client.projects.projects import Project
from gretel_client.projects.records import RecordHandler

import gretel_trainer.b2.status as status
from gretel_trainer.b2.core import BenchmarkException, Dataset, RunIdentifier, Timer
from gretel_trainer.b2.gretel_models import GretelModel

logger = logging.getLogger(__name__)


class GretelSDKExecutor:
    def __init__(
        self,
        project: Project,
        benchmark_model: GretelModel,
        run_identifier: RunIdentifier,
        statuses: DictProxy,
        refresh_interval: int,
    ):
        self.project = project
        self.benchmark_model = benchmark_model
        self.refresh_interval = refresh_interval
        self.run_identifier = run_identifier
        self.statuses = statuses
        self.set_status(status.NotStarted())

        self.model: Optional[Model] = None
        self.source: Optional[Dataset] = None
        self.record_handler: Optional[RecordHandler] = None
        self.train_time: Optional[float] = None
        self.generate_time: Optional[float] = None

    def set_status(self, status: status.Status) -> None:
        self.status = status
        self.statuses[self.run_identifier] = status

    def train(self, source: Dataset) -> None:
        # TODO: potentially skip (datatype, col count, etc.)
        logger.info(f"Starting training of {self.benchmark_model.name} on {source.name}")
        self.set_status(status.InProgress(stage="train"))
        self.source = source
        self.model = self.project.create_model_obj(
            model_config=self.benchmark_model.config, data_source=source
        )
        train_time = Timer()
        with train_time:
            self.model.submit_cloud()
            job_status = _await_job(self.model, self.refresh_interval)
        self.train_time = train_time.duration()
        if job_status in END_STATES and job_status != Status.COMPLETED:
            self.set_status(status.Failed(during="train", train_secs=self.train_time))

    def generate(self) -> None:
        if self.model is None or self.source is None or self.train_time is None:
            raise BenchmarkException("Cannot generate before training")

        if isinstance(self.status, status.Failed):
            return None

        logger.info(f"Starting synthetic data generation for {self.benchmark_model.name} trained on {self.source.name}")
        self.set_status(status.InProgress(stage="generate", train_secs=self.train_time))
        self.record_handler = self.model.create_record_handler_obj(
            params={"num_records": self.source.row_count}
        )
        generate_time = Timer()
        with generate_time:
            self.record_handler.submit_cloud()
            job_status = _await_job(self.record_handler, self.refresh_interval)
        self.generate_time = generate_time.duration()
        if job_status == Status.COMPLETED:
            self.set_status(status.Completed(
                sqs=self.get_sqs_score(),
                train_secs=self.train_time,
                generate_secs=self.generate_time,
                synthetic_data=self.get_synthetic_data(),
            ))
        elif job_status in END_STATES:
            self.set_status(status.Failed(
                during="train",
                train_secs=self.train_time,
                generate_secs=self.generate_time,
            ))

    def get_sqs_score(self) -> int:
        if self.model is None:
            raise BenchmarkException("Cannot get SQS score before training")

        report = self.model.peek_report()
        return report["synthetic_data_quality_score"]["score"]

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
