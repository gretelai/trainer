import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import smart_open
from gretel_client.projects.jobs import ACTIVE_STATES, END_STATES, Job, Status
from gretel_client.projects.models import Model, read_model_config
from gretel_client.projects.projects import Project
from gretel_client.projects.records import RecordHandler

from gretel_trainer.b2.core import (
    BenchmarkException,
    Dataset,
    RunIdentifier,
    log,
    run_out_path,
)
from gretel_trainer.b2.gretel.models import GretelModel, GretelModelConfig


class GretelSDKStrategy:
    def __init__(
        self,
        benchmark_model: GretelModel,
        dataset: Dataset,
        run_identifier: RunIdentifier,
        project: Project,
        refresh_interval: int,
        working_dir: Path,
    ):
        self.benchmark_model = benchmark_model
        self.dataset = dataset
        self.run_identifier = run_identifier
        self.project = project
        self.refresh_interval = refresh_interval
        self.working_dir = working_dir

        self.model: Optional[Model] = None
        self.record_handler: Optional[RecordHandler] = None
        self.evaluate_report_json: Optional[Dict[str, Any]] = None

    def _format_model_config(self) -> GretelModelConfig:
        config = read_model_config(self.benchmark_model.config)
        config["name"] = str(self.run_identifier)
        return config

    def runnable(self, dataset: Dataset) -> bool:
        return self.benchmark_model.runnable(dataset)

    def get_train_time(self) -> Optional[float]:
        return _get_duration(self.model)

    def get_generate_time(self) -> Optional[float]:
        return _get_duration(self.record_handler)

    def train(self) -> None:
        model_config = self._format_model_config()
        self.model = self.project.create_model_obj(
            model_config=model_config, data_source=self.dataset.data_source
        )
        self.model.submit_cloud()
        job_status = self._await_job(self.model, "training")
        if job_status in END_STATES and job_status != Status.COMPLETED:
            raise BenchmarkException("Training failed")

    def generate(self) -> None:
        if self.model is None:
            raise BenchmarkException("Cannot generate before training")

        self.record_handler = self.model.create_record_handler_obj(
            params={"num_records": self.dataset.row_count}
        )
        self.record_handler.submit_cloud()
        job_status = self._await_job(self.record_handler, "generation")
        if job_status == Status.COMPLETED:
            synthetic_data = pd.read_csv(
                self.record_handler.get_artifact_link("data"), compression="gzip"
            )
            synthetic_data.to_csv(self._synthetic_data_path, index=False)
        else:
            raise BenchmarkException("Generate failed")

    def evaluate(self) -> None:
        evaluate_model = self.project.create_model_obj(
            model_config="evaluate/default",
            data_source=str(self._synthetic_data_path),
            ref_data=self.dataset.data_source,
        )
        evaluate_model.submit_cloud()
        job_status = self._await_job(evaluate_model, "evaluation")
        if job_status in END_STATES and job_status != Status.COMPLETED:
            raise BenchmarkException("Evaluate failed")
        self.evaluate_report_json = json.loads(
            smart_open.open(evaluate_model.get_artifact_link("report_json")).read()
        )

    def get_sqs_score(self) -> Optional[int]:
        if self.evaluate_report_json is None:
            return None
        else:
            return self.evaluate_report_json["synthetic_data_quality_score"]["score"]

    @property
    def _synthetic_data_path(self) -> Path:
        return run_out_path(self.working_dir, self.run_identifier)

    def _await_job(self, job: Job, task: str) -> Status:
        failed_refresh_attempts = 0
        status = job.status
        while not _finished(status):
            if failed_refresh_attempts >= 5:
                return Status.LOST

            time.sleep(self.refresh_interval)

            status, failed_refresh_attempts = _cautiously_refresh_status(
                job, failed_refresh_attempts
            )
            self._log_in_progress(status, task)
        return status

    def _log_in_progress(self, status: Status, task: str) -> None:
        if status in ACTIVE_STATES:
            log(self.run_identifier, f"{task} job still in progress (status: {status})")


def _get_duration(job: Optional[Job]) -> Optional[float]:
    if job is None:
        return None
    else:
        return job.billing_details["total_time_seconds"]


def _finished(status: Status) -> bool:
    return status in END_STATES


def _cautiously_refresh_status(job: Job, attempts: int) -> Tuple[Status, int]:
    try:
        job.refresh()
        attempts = 0
    except:
        attempts = attempts + 1

    return (job.status, attempts)
