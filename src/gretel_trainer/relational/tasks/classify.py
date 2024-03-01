import shutil

import gretel_trainer.relational.tasks.common as common

from gretel_client.projects.artifact_handlers import open_artifact
from gretel_client.projects.jobs import Job
from gretel_client.projects.models import Model
from gretel_client.projects.records import RecordHandler
from gretel_trainer.relational.output_handler import OutputHandler
from gretel_trainer.relational.task_runner import TaskContext
from gretel_trainer.relational.workflow_state import Classify


class ClassifyTask:
    def __init__(
        self,
        classify: Classify,
        data_sources: dict[str, str],
        all_rows: bool,
        ctx: TaskContext,
        output_handler: OutputHandler,
    ):
        self.classify = classify
        self.data_sources = data_sources
        self.all_rows = all_rows
        self.ctx = ctx
        self.output_handler = output_handler
        self.classify_record_handlers: dict[str, RecordHandler] = {}
        self.completed_models = []
        self.failed_models = []
        self.completed_record_handlers = []
        self.failed_record_handlers = []
        self.result_filepaths: dict[str, str] = {}

    def action(self, job: Job) -> str:
        if self.all_rows:
            if isinstance(job, Model):
                return "classify training"
            else:
                return "classification (all rows)"
        else:
            return "classification"

    @property
    def table_collection(self) -> list[str]:
        return list(self.classify.models.keys())

    def more_to_do(self) -> bool:
        total_tables = len(self.classify.models)
        any_unfinished_models = len(self._finished_models) < total_tables
        any_unfinished_record_handlers = (
            len(self._finished_record_handlers) < total_tables
        )

        if self.all_rows:
            return any_unfinished_models or any_unfinished_record_handlers
        else:
            return any_unfinished_models

    @property
    def _finished_models(self) -> list[str]:
        return self.completed_models + self.failed_models

    @property
    def _finished_record_handlers(self) -> list[str]:
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
        model = self.classify.models.get(table)
        return record_handler or model

    def handle_completed(self, table: str, job: Job) -> None:
        if isinstance(job, Model):
            self.completed_models.append(table)
            common.log_success(table, self.action(job))
            if self.all_rows:
                record_handler = job.create_record_handler_obj(
                    data_source=self.data_sources[table]
                )
                self.classify_record_handlers[table] = record_handler
                self.ctx.maybe_start_job(
                    job=record_handler, table_name=table, action=self.action(job)
                )
        elif isinstance(job, RecordHandler):
            self.completed_record_handlers.append(table)
            common.log_success(table, self.action(job))
        self._write_results(job=job, table=table)
        common.cleanup(sdk=self.ctx.extended_sdk, project=self.ctx.project, job=job)

    def handle_failed(self, table: str, job: Job) -> None:
        if isinstance(job, Model):
            self.failed_models.append(table)
        elif isinstance(job, RecordHandler):
            self.failed_record_handlers.append(table)
        common.log_failed(table, self.action(job))
        common.cleanup(sdk=self.ctx.extended_sdk, project=self.ctx.project, job=job)

    def handle_lost_contact(self, table: str, job: Job) -> None:
        if isinstance(job, Model):
            self.failed_models.append(table)
        elif isinstance(job, RecordHandler):
            self.failed_record_handlers.append(table)
        common.log_lost_contact(table)
        common.cleanup(sdk=self.ctx.extended_sdk, project=self.ctx.project, job=job)

    def handle_in_progress(self, table: str, job: Job) -> None:
        common.log_in_progress(table, job.status, self.action(job))

    def each_iteration(self) -> None:
        self.ctx.backup()

    def _write_results(self, job: Job, table: str) -> None:
        if isinstance(job, Model):
            filename = f"classify_{table}.gz"
            artifact_name = "data_preview"
        else:
            filename = f"classify_all_rows_{table}.gz"
            artifact_name = "data"

        destpath = self.output_handler.filepath_for(filename)

        with job.get_artifact_handle(artifact_name) as src, open_artifact(
            str(destpath), "wb"
        ) as dest:
            shutil.copyfileobj(src, dest)
        self.result_filepaths[table] = destpath
