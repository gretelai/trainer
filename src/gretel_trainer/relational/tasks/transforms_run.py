from typing import Dict, List, Optional

import pandas as pd
from gretel_client.projects.jobs import Job
from gretel_client.projects.projects import Project
from gretel_client.projects.records import RecordHandler

from gretel_trainer.relational.sdk_extras import get_record_handler_data
from gretel_trainer.relational.tasks.common import _MultiTable


class TransformsRunTask:
    def __init__(
        self,
        record_handlers: Dict[str, RecordHandler],
        multitable: _MultiTable,
    ):
        self.record_handlers = record_handlers
        self.multitable = multitable
        self.working_tables: Dict[str, Optional[pd.DataFrame]] = {}

    @property
    def output_tables(self) -> Dict[str, pd.DataFrame]:
        return {
            table: data
            for table, data in self.working_tables.items()
            if data is not None
        }

    @property
    def action(self) -> str:
        return "transforms run"

    @property
    def refresh_interval(self) -> int:
        return self.multitable._refresh_interval

    @property
    def project(self) -> Project:
        return self.multitable._project

    @property
    def table_collection(self) -> List[str]:
        return list(self.record_handlers.keys())

    @property
    def artifacts_per_job(self) -> int:
        return 1

    def more_to_do(self) -> bool:
        return len(self.working_tables) < len(self.record_handlers)

    def is_finished(self, table: str) -> bool:
        return table in self.working_tables

    def get_job(self, table: str) -> Job:
        return self.record_handlers[table]

    def handle_completed(self, table: str, job: Job) -> None:
        self.working_tables[table] = get_record_handler_data(job)

    def handle_failed(self, table: str) -> None:
        self.working_tables[table] = None

    def handle_lost_contact(self, table: str) -> None:
        self.working_tables[table] = None

    def each_iteration(self) -> None:
        pass
