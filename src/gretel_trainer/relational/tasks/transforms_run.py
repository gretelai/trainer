from typing import Optional

import pandas as pd
from gretel_client.projects.jobs import Job
from gretel_client.projects.projects import Project
from gretel_client.projects.records import RecordHandler

import gretel_trainer.relational.tasks.common as common

ACTION = "transforms run"


class TransformsRunTask:
    def __init__(
        self,
        record_handlers: dict[str, RecordHandler],
        multitable: common._MultiTable,
    ):
        self.record_handlers = record_handlers
        self.multitable = multitable
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
    def project(self) -> Project:
        return self.multitable._project

    @property
    def table_collection(self) -> list[str]:
        return list(self.record_handlers.keys())

    @property
    def artifacts_per_job(self) -> int:
        return 1

    def more_to_do(self) -> bool:
        return len(self.working_tables) < len(self.record_handlers)

    def wait(self) -> None:
        common.wait(self.multitable._refresh_interval)

    def is_finished(self, table: str) -> bool:
        return table in self.working_tables

    def get_job(self, table: str) -> Job:
        return self.record_handlers[table]

    def handle_completed(self, table: str, job: Job) -> None:
        self.working_tables[
            table
        ] = self.multitable._extended_sdk.get_record_handler_data(job)
        common.log_success(table, ACTION)
        common.cleanup(sdk=self.multitable._extended_sdk, project=self.project, job=job)

    def handle_failed(self, table: str, job: Job) -> None:
        self.working_tables[table] = None
        common.log_failed(table, ACTION)
        common.cleanup(sdk=self.multitable._extended_sdk, project=self.project, job=job)

    def handle_lost_contact(self, table: str, job: Job) -> None:
        self.working_tables[table] = None
        common.log_lost_contact(table)
        common.cleanup(sdk=self.multitable._extended_sdk, project=self.project, job=job)

    def handle_in_progress(self, table: str, job: Job) -> None:
        common.log_in_progress(table, job.status, ACTION)

    def each_iteration(self) -> None:
        pass
