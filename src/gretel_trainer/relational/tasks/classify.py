from typing import List

from gretel_client.projects.jobs import Job
from gretel_client.projects.models import Model
from gretel_client.projects.projects import Project
from gretel_client.projects.records import RecordHandler

import gretel_trainer.relational.tasks.common as common
from gretel_trainer.relational.workflow_state import Classify


class ClassifyTask:
    def __init__(
        self,
        classify: Classify,
        multitable: common._MultiTable,
    ):
        self.classify = classify
        self.multitable = multitable
        self.completed_models = []
        self.failed_models = []
        self.completed_record_handlers = []
        self.failed_record_handlers = []

    def action(self, job: Job) -> str:
        if isinstance(job, Model):
            return "classify training"
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
        return list(self.classify.models.keys())

    def more_to_do(self) -> bool:
        total_tables = len(self.classify.models)
        return (
            len(self._finished_models) < total_tables
            or len(self._finished_record_handlers) < total_tables
        )

    def wait(self) -> None:
        # Classify training is fast
        if len(self._finished_models) < len(self.classify.models):
            duration = 15
        else:
            duration = self.multitable._refresh_interval
        common.wait(duration)

    @property
    def _finished_models(self) -> List[str]:
        return self.completed_models + self.failed_models

    @property
    def _finished_record_handlers(self) -> List[str]:
        return self.completed_record_handlers + self.failed_record_handlers

    def is_finished(self, table: str) -> bool:
        return (
            table in self._finished_models and table in self._finished_record_handlers
        )

    def get_job(self, table: str) -> Job:
        record_handler = self.classify.record_handlers.get(table)
        model = self.classify.models.get(table)
        return record_handler or model

    def handle_completed(self, table: str, job: Job) -> None:
        if isinstance(job, Model):
            self.completed_models.append(table)
            common.log_success(table, self.action(job))
            record_handler = job.create_record_handler_obj(data_source=job.data_source)
            self.classify.record_handlers[table] = record_handler
            self.multitable._extended_sdk.start_job_if_possible(
                job=record_handler,
                table_name=table,
                action=self.action(record_handler),
                project=self.project,
                number_of_artifacts=self.artifacts_per_job,
            )
        elif isinstance(job, RecordHandler):
            self.completed_record_handlers.append(table)
            common.log_success(table, self.action(job))
            common.cleanup(
                sdk=self.multitable._extended_sdk, project=self.project, job=job
            )

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
            self.classify.lost_models.append(table)
        elif isinstance(job, RecordHandler):
            self.failed_record_handlers.append(table)
            self.classify.lost_record_handlers.append(table)
        common.log_lost_contact(table)
        common.cleanup(sdk=self.multitable._extended_sdk, project=self.project, job=job)

    def handle_in_progress(self, table: str, job: Job) -> None:
        action = self.action(job)
        common.log_in_progress(table, job.status, action)

    def each_iteration(self) -> None:
        self.multitable._backup()
