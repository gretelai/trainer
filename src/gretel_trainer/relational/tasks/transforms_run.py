from typing import Optional

import pandas as pd

import gretel_trainer.relational.tasks.common as common

from gretel_client.projects.jobs import Job
from gretel_client.projects.records import RecordHandler
from gretel_trainer.relational.task_runner import TaskContext

ACTION = "transforms run"


class TransformsRunTask:
    def __init__(
        self,
        record_handlers: dict[str, RecordHandler],
        ctx: TaskContext,
    ):
        self.record_handlers = record_handlers
        self.ctx = ctx
        self.working_tables: dict[str, Optional[pd.DataFrame]] = {}

    @property
    def output_tables(self) -> dict[str, pd.DataFrame]:
        return {
            table: data
            for table, data in self.working_tables.items()
            if data is not None
        }

    def action(self, job: Job) -> str:
        return ACTION

    @property
    def table_collection(self) -> list[str]:
        return list(self.record_handlers.keys())

    def more_to_do(self) -> bool:
        return len(self.working_tables) < len(self.record_handlers)

    def is_finished(self, table: str) -> bool:
        return table in self.working_tables

    def get_job(self, table: str) -> Job:
        return self.record_handlers[table]

    def handle_completed(self, table: str, job: Job) -> None:
        self.working_tables[table] = self.ctx.extended_sdk.get_record_handler_data(job)
        common.log_success(table, self.action(job))
        common.cleanup(sdk=self.ctx.extended_sdk, project=self.ctx.project, job=job)

    def handle_failed(self, table: str, job: Job) -> None:
        self.working_tables[table] = None
        common.log_failed(table, self.action(job))
        common.cleanup(sdk=self.ctx.extended_sdk, project=self.ctx.project, job=job)

    def handle_lost_contact(self, table: str, job: Job) -> None:
        self.working_tables[table] = None
        common.log_lost_contact(table)
        common.cleanup(sdk=self.ctx.extended_sdk, project=self.ctx.project, job=job)

    def handle_in_progress(self, table: str, job: Job) -> None:
        common.log_in_progress(table, job.status, self.action(job))

    def each_iteration(self) -> None:
        pass
