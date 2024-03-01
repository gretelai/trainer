import logging

from typing import Optional, Union

import pandas as pd

import gretel_trainer.relational.tasks.common as common

from gretel_client.projects.jobs import ACTIVE_STATES, Job, Status
from gretel_client.projects.records import RecordHandler
from gretel_trainer.relational.core import RelationalData
from gretel_trainer.relational.output_handler import OutputHandler
from gretel_trainer.relational.strategies.ancestral import AncestralStrategy
from gretel_trainer.relational.strategies.independent import IndependentStrategy
from gretel_trainer.relational.task_runner import TaskContext
from gretel_trainer.relational.workflow_state import SyntheticsRun, SyntheticsTrain

logger = logging.getLogger(__name__)

ACTION = "synthetic data generation"


class SyntheticsRunTask:
    def __init__(
        self,
        synthetics_run: SyntheticsRun,
        synthetics_train: SyntheticsTrain,
        subdir: str,
        output_handler: OutputHandler,
        rel_data: RelationalData,
        strategy: Union[IndependentStrategy, AncestralStrategy],
        ctx: TaskContext,
    ):
        self.synthetics_run = synthetics_run
        self.synthetics_train = synthetics_train
        self.subdir = subdir
        self.output_handler = output_handler
        self.rel_data = rel_data
        self.strategy = strategy
        self.ctx = ctx
        self.working_tables = self._setup_working_tables()

    def _setup_working_tables(self) -> dict[str, Optional[pd.DataFrame]]:
        working_tables = {}
        all_tables = self.rel_data.list_all_tables()

        for table in all_tables:
            if table in self.synthetics_train.bypass:
                source_row_count = self.rel_data.get_table_row_count(table)
                out_row_count = int(
                    source_row_count * self.synthetics_run.record_size_ratio
                )
                working_tables[table] = pd.DataFrame(index=range(out_row_count))
                continue

            model = self.synthetics_train.models.get(table)

            # Table was either omitted from training or marked as to-be-preserved during generation
            if model is None or table in self.synthetics_run.preserved:
                working_tables[table] = self.strategy.get_preserved_data(
                    table, self.rel_data
                )
                continue

            # Table was included in training, but failed at that step
            if model.status != Status.COMPLETED:
                working_tables[table] = None
                continue

        return working_tables

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
        return list(self.synthetics_run.record_handlers.keys())

    def more_to_do(self) -> bool:
        return len(self.working_tables) < len(self._all_tables)

    def is_finished(self, table: str) -> bool:
        return table in self.working_tables

    def get_job(self, table: str) -> Job:
        return self.synthetics_run.record_handlers[table]

    def handle_completed(self, table: str, job: Job) -> None:
        record_handler_data = self.ctx.extended_sdk.get_record_handler_data(job)
        post_processed_data = self.strategy.post_process_individual_synthetic_result(
            table_name=table,
            rel_data=self.rel_data,
            synthetic_table=record_handler_data,
            record_size_ratio=self.synthetics_run.record_size_ratio,
        )
        self.working_tables[table] = post_processed_data
        common.log_success(table, self.action(job))
        common.cleanup(sdk=self.ctx.extended_sdk, project=self.ctx.project, job=job)

    def handle_failed(self, table: str, job: Job) -> None:
        self.working_tables[table] = None
        self._fail_table(table)
        common.log_failed(table, self.action(job))
        common.cleanup(sdk=self.ctx.extended_sdk, project=self.ctx.project, job=job)
        self.ctx.backup()

    def handle_lost_contact(self, table: str, job: Job) -> None:
        self.synthetics_run.lost_contact.append(table)
        self._fail_table(table)
        common.log_lost_contact(table)
        common.cleanup(sdk=self.ctx.extended_sdk, project=self.ctx.project, job=job)
        self.ctx.backup()

    def handle_in_progress(self, table: str, job: Job) -> None:
        common.log_in_progress(table, job.status, self.action(job))

    def each_iteration(self) -> None:
        # Determine if we can start any more jobs
        in_progress_tables = [
            table
            for table in self._all_tables
            if _table_is_in_progress(self.synthetics_run.record_handlers, table)
        ]
        finished_tables = [table for table in self.working_tables]

        ready_tables = self.strategy.ready_to_generate(
            self.rel_data, in_progress_tables, finished_tables
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

            table_job = self.strategy.get_generation_job(
                table_name,
                self.rel_data,
                self.synthetics_run.record_size_ratio,
                present_working_tables,
                self.subdir,
                self.output_handler,
            )
            model = self.synthetics_train.models[table_name]
            record_handler = model.create_record_handler_obj(**table_job)
            self.synthetics_run.record_handlers[table_name] = record_handler
            # Attempt starting the record handler right away. If it can't start right at this moment,
            # the regular task runner check will handle starting it when possible.
            self.ctx.maybe_start_job(
                job=record_handler,
                table_name=table_name,
                action=self.action(record_handler),
            )

        self.ctx.backup()

    @property
    def _all_tables(self) -> list[str]:
        return self.rel_data.list_all_tables()

    def _fail_table(self, table: str) -> None:
        self.working_tables[table] = None
        for other_table in self.strategy.tables_to_skip_when_failed(
            table, self.rel_data
        ):
            _log_skipping(skip=other_table, failed_parent=table)
            self.working_tables[other_table] = None


def _log_skipping(skip: str, failed_parent: str) -> None:
    logger.info(
        f"Skipping synthetic data generation for `{skip}` because it depends on `{failed_parent}`"
    )


def _table_is_in_progress(
    record_handlers: dict[str, RecordHandler], table: str
) -> bool:
    in_progress = False

    record_handler = record_handlers.get(table)
    if record_handler is not None and record_handler.record_id is not None:
        in_progress = record_handler.status in ACTIVE_STATES

    return in_progress
