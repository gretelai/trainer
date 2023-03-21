import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from gretel_client.projects.jobs import ACTIVE_STATES, Job
from gretel_client.projects.models import Model
from gretel_client.projects.projects import Project
from gretel_client.projects.records import RecordHandler

from gretel_trainer.relational.sdk_extras import (
    get_record_handler_data,
    start_job_if_possible,
)
from gretel_trainer.relational.tasks.common import _MultiTable
from gretel_trainer.relational.workflow_state import SyntheticsRun, SyntheticsTrain

logger = logging.getLogger(__name__)


class SyntheticsRunTask:
    def __init__(
        self,
        synthetics_run: SyntheticsRun,
        synthetics_train: SyntheticsTrain,
        run_dir: Path,
        multitable: _MultiTable,
    ):
        self.synthetics_run = synthetics_run
        self.synthetics_train = synthetics_train
        self.run_dir = run_dir
        self.multitable = multitable
        self.working_tables = self._setup_working_tables()

    def _setup_working_tables(self) -> Dict[str, Optional[pd.DataFrame]]:
        working_tables = {}

        for table in self.synthetics_run.missing_model:
            working_tables[table] = None

        for table in self.synthetics_run.preserved:
            working_tables[table] = self.multitable._strategy.get_preserved_data(
                table, self.multitable.relational_data
            )

        return working_tables

    @property
    def output_tables(self) -> Dict[str, pd.DataFrame]:
        return {
            table: data
            for table, data in self.working_tables.items()
            if data is not None
        }

    @property
    def action(self) -> str:
        return "synthetic data generation"

    @property
    def refresh_interval(self) -> int:
        return self.multitable._refresh_interval

    @property
    def project(self) -> Project:
        return self.multitable._project

    @property
    def table_collection(self) -> List[str]:
        return list(self.synthetics_run.record_handlers.keys())

    @property
    def artifacts_per_job(self) -> int:
        return 1

    def more_to_do(self) -> bool:
        return len(self.working_tables) < len(self._all_tables)

    def is_finished(self, table: str) -> bool:
        return table in self.working_tables

    def get_job(self, table: str) -> Job:
        return self.synthetics_run.record_handlers[table]

    def handle_completed(self, table: str, job: Job) -> None:
        record_handler_data = get_record_handler_data(job)
        post_processed_data = (
            self.multitable._strategy.post_process_individual_synthetic_result(
                table, self.multitable.relational_data, record_handler_data
            )
        )
        self.working_tables[table] = post_processed_data

    def handle_failed(self, table: str) -> None:
        self.working_tables[table] = None
        self._fail_table(table)
        self.multitable._backup()

    def handle_lost_contact(self, table: str) -> None:
        self.synthetics_run.lost_contact.append(table)
        self._fail_table(table)
        self.multitable._backup()

    def each_iteration(self) -> None:
        # Determine if we can start any more jobs
        in_progress_tables = [
            table
            for table in self._all_tables
            if _table_is_in_progress(self.synthetics_run.record_handlers, table)
        ]
        finished_tables = [table for table in self.working_tables]

        ready_tables = self.multitable._strategy.ready_to_generate(
            self.multitable.relational_data, in_progress_tables, finished_tables
        )

        for table_name in ready_tables:
            # Any record handlers we created but deferred submitting will continue to register as "ready" until they are actually submitted and become "in progress".
            # This check prevents repeatedly incurring the cost of fetching the job details (and logging duplicatively) while the job is deferred.
            if self.synthetics_run.record_handlers.get(table_name) is not None:
                continue

            present_working_tables = {
                table: data
                for table, data in self.working_tables.items()
                if data is not None
            }

            table_job = self.multitable._strategy.get_generation_job(
                table_name,
                self.multitable.relational_data,
                self.synthetics_run.record_size_ratio,
                present_working_tables,
                self.run_dir,
                self.synthetics_train.training_columns[table_name],
            )
            model = self.synthetics_train.models[table_name]
            record_handler = model.create_record_handler_obj(**table_job)
            self.synthetics_run.record_handlers[table_name] = record_handler
            # Attempt starting the record handler right away. If it can't start right at this moment,
            # the regular task runner check will handle starting it when possible.
            start_job_if_possible(
                job=record_handler,
                table_name=table_name,
                action=self.action,
                project=self.project,
                number_of_artifacts=self.artifacts_per_job,
            )

        self.multitable._backup()

    @property
    def _all_tables(self) -> List[str]:
        return self.multitable.relational_data.list_all_tables()

    def _fail_table(self, table: str) -> None:
        self.working_tables[table] = None
        for other_table in self.multitable._strategy.tables_to_skip_when_failed(
            table, self.multitable.relational_data
        ):
            _log_skipping(skip=other_table, failed_parent=table)
            self.working_tables[other_table] = None


def _log_skipping(skip: str, failed_parent: str) -> None:
    logger.info(
        f"Skipping synthetic data generation for `{skip}` because it depends on `{failed_parent}`"
    )


def _table_is_in_progress(
    record_handlers: Dict[str, RecordHandler], table: str
) -> bool:
    in_progress = False

    record_handler = record_handlers.get(table)
    if record_handler is not None and record_handler.record_id is not None:
        in_progress = record_handler.status in ACTIVE_STATES

    return in_progress
