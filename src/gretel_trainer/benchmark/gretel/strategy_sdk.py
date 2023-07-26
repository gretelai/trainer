import copy
import gzip
from pathlib import Path
from typing import Optional

import requests
from gretel_client.projects.jobs import END_STATES, Job, RunnerMode, Status
from gretel_client.projects.models import Model, read_model_config
from gretel_client.projects.projects import Project
from gretel_client.projects.records import RecordHandler

from gretel_trainer.benchmark.core import (
    BenchmarkConfig,
    BenchmarkException,
    Dataset,
    run_out_path,
)
from gretel_trainer.benchmark.gretel.models import GretelModel, GretelModelConfig
from gretel_trainer.benchmark.job_spec import JobSpec
from gretel_trainer.benchmark.sdk_extras import await_job


class GretelSDKStrategy:
    def __init__(
        self,
        job_spec: JobSpec[GretelModel],
        artifact_key: Optional[str],
        run_identifier: str,
        project: Project,
        config: BenchmarkConfig,
    ):
        self.job_spec = job_spec
        self.artifact_key = artifact_key
        self.run_identifier = run_identifier
        self.project = project
        self.config = config

        self.model: Optional[Model] = None
        self.record_handler: Optional[RecordHandler] = None

    @property
    def dataset(self) -> Dataset:
        return self.job_spec.dataset

    @property
    def benchmark_model(self) -> GretelModel:
        return self.job_spec.model

    def _format_model_config(self) -> GretelModelConfig:
        config = copy.deepcopy(read_model_config(self.benchmark_model.config))
        config["name"] = self.run_identifier
        return config

    def runnable(self) -> bool:
        return self.benchmark_model.runnable(self.dataset)

    def get_train_time(self) -> Optional[float]:
        return _get_duration(self.model)

    def get_generate_time(self) -> Optional[float]:
        return _get_duration(self.record_handler)

    def train(self) -> None:
        model_config = self._format_model_config()
        data_source = self.artifact_key or self.dataset.data_source
        self.model = self.project.create_model_obj(
            model_config=model_config, data_source=data_source
        )
        # Calling this in lieu of submit_cloud() is supposed to avoid
        # artifact upload. Doesn't work for more recent client versions!
        self.model.submit(runner_mode=RunnerMode.CLOUD)
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
            self._download_synthetic_data(self.record_handler)
        else:
            raise BenchmarkException("Generate failed")

    def _download_synthetic_data(self, record_handler: RecordHandler) -> None:
        response = self._get_record_handler_data(record_handler)
        with open(self._synthetic_data_path, "wb") as out:
            out.write(gzip.decompress(response.content))

    def _get_record_handler_data(
        self, record_handler: RecordHandler
    ) -> requests.Response:
        artifact_link = record_handler.get_artifact_link("data")
        return requests.get(artifact_link)

    @property
    def evaluate_ref_data(self) -> str:
        if self.artifact_key is not None:
            return self.artifact_key
        if self.model is not None:
            return self.model.data_source
        return self.dataset.data_source

    @property
    def _synthetic_data_path(self) -> Path:
        return run_out_path(self.config.working_dir, self.run_identifier)

    def _await_job(self, job: Job, task: str) -> Status:
        return await_job(self.run_identifier, job, task, self.config.refresh_interval)


def _get_duration(job: Optional[Job]) -> Optional[float]:
    if job is None:
        return None
    else:
        return job.billing_details.get("total_time_seconds")
