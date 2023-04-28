import shutil
from pathlib import Path
from typing import Dict, List

import smart_open
from gretel_client.projects.jobs import Job
from gretel_client.projects.models import Model
from gretel_client.projects.projects import Project
from gretel_client.projects.records import RecordHandler

import gretel_trainer.relational.tasks.common as common


class ClassifyTask:
    def __init__(
        self,
        classify_models: Dict[str, Model],
        all_rows: bool,
        multitable: common._MultiTable,
        out_dir: Path,
    ):
        self.classify_models = classify_models
        self.classify_record_handlers: Dict[str, RecordHandler] = {}
        self.all_rows = all_rows
        self.multitable = multitable
        self.out_dir = out_dir
        self.completed_models = []
        self.failed_models = []
        self.completed_record_handlers = []
        self.failed_record_handlers = []
        self.result_filepaths: Dict[str, Path] = {}

    def action(self, job: Job) -> str:
        if self.all_rows:
            if isinstance(job, Model):
                return "classify training"
            else:
                return "classification (all rows)"
        else:
            return "classification"

    @property
    def project(self) -> Project:
        return self.multitable._project

    @property
    def artifacts_per_job(self) -> int:
        return 1

    @property
    def table_collection(self) -> List[str]:
        return list(self.classify_models.keys())

    def more_to_do(self) -> bool:
        total_tables = len(self.classify_models)
        any_unfinished_models = len(self._finished_models) < total_tables
        any_unfinished_record_handlers = (
            len(self._finished_record_handlers) < total_tables
        )

        if self.all_rows:
            return any_unfinished_models or any_unfinished_record_handlers
        else:
            return any_unfinished_models

    def wait(self) -> None:
        if self.all_rows:
            duration = self.multitable._refresh_interval
        else:
            duration = 15
        common.wait(duration)

    @property
    def _finished_models(self) -> List[str]:
        return self.completed_models + self.failed_models

    @property
    def _finished_record_handlers(self) -> List[str]:
        return self.completed_record_handlers + self.failed_record_handlers

    def is_finished(self, table: str) -> bool:
        if self.all_rows:
            return (
                table in self._finished_models
                and table in self._finished_record_handlers
            )
        else:
            return table in self._finished_models

    def get_job(self, table: str) -> Job:
        record_handler = self.classify_record_handlers.get(table)
        model = self.classify_models.get(table)
        return record_handler or model

    def handle_completed(self, table: str, job: Job) -> None:
        if isinstance(job, Model):
            self.completed_models.append(table)
            common.log_success(table, self.action(job))
            if self.all_rows:
                record_handler = job.create_record_handler_obj(
                    data_source=job.data_source
                )
                self.classify_record_handlers[table] = record_handler
                self.multitable._extended_sdk.start_job_if_possible(
                    job=record_handler,
                    table_name=table,
                    action=self.action(record_handler),
                    project=self.project,
                    number_of_artifacts=self.artifacts_per_job,
                )
            else:
                common.cleanup(
                    sdk=self.multitable._extended_sdk, project=self.project, job=job
                )
        elif isinstance(job, RecordHandler):
            self.completed_record_handlers.append(table)
            common.log_success(table, self.action(job))
            common.cleanup(
                sdk=self.multitable._extended_sdk, project=self.project, job=job
            )
        self._write_results(job=job, table=table)

    def handle_failed(self, table: str, job: Job) -> None:
        if isinstance(job, Model):
            self.failed_models.append(table)
        elif isinstance(job, RecordHandler):
            self.failed_record_handlers.append(table)
        common.log_failed(table, self.action(job))
        common.cleanup(sdk=self.multitable._extended_sdk, project=self.project, job=job)

    def handle_lost_contact(self, table: str, job: Job) -> None:
        if isinstance(job, Model):
            self.failed_models.append(table)
        elif isinstance(job, RecordHandler):
            self.failed_record_handlers.append(table)
        common.log_lost_contact(table)
        common.cleanup(sdk=self.multitable._extended_sdk, project=self.project, job=job)

    def handle_in_progress(self, table: str, job: Job) -> None:
        action = self.action(job)
        common.log_in_progress(table, job.status, action)

    def each_iteration(self) -> None:
        self.multitable._backup()

    def _write_results(self, job: Job, table: str) -> None:
        if isinstance(job, Model):
            filename = f"classify_subset_{table}.gz"
            artifact_name = "data_preview"
        else:
            filename = f"classify_all_rows_{table}.gz"
            artifact_name = "data"

        destpath = self.out_dir / filename

        with smart_open.open(
            job.get_artifact_link(artifact_name), "rb"
        ) as src, smart_open.open(str(destpath), "wb") as dest:
            shutil.copyfileobj(src, dest)
        self.result_filepaths[filename] = destpath
