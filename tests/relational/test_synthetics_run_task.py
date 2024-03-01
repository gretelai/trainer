from typing import Optional
from unittest.mock import Mock, patch

import pandas as pd
import pandas.testing as pdtest

from gretel_client.projects.jobs import Status
from gretel_trainer.relational.core import RelationalData
from gretel_trainer.relational.output_handler import OutputHandler
from gretel_trainer.relational.sdk_extras import ExtendedGretelSDK, MAX_IN_FLIGHT_JOBS
from gretel_trainer.relational.strategies.ancestral import AncestralStrategy
from gretel_trainer.relational.task_runner import TaskContext
from gretel_trainer.relational.tasks.synthetics_run import SyntheticsRunTask
from gretel_trainer.relational.workflow_state import SyntheticsRun, SyntheticsTrain


class MockStrategy(AncestralStrategy):
    def post_process_individual_synthetic_result(
        self, table_name, rel_data, synthetic_table, record_size_ratio
    ):
        return synthetic_table.head(1)


def make_task(
    rel_data: RelationalData,
    output_handler: OutputHandler,
    preserved: Optional[list[str]] = None,
    failed: Optional[list[str]] = None,
    omitted: Optional[list[str]] = None,
) -> SyntheticsRunTask:
    def _status_for_table(table: str, failed: list[str]) -> Status:
        if table in failed:
            return Status.ERROR
        else:
            return Status.COMPLETED

    context = TaskContext(
        in_flight_jobs=0,
        refresh_interval=0,
        project=Mock(),
        extended_sdk=ExtendedGretelSDK(hybrid=False),
        backup=lambda: None,
    )
    return SyntheticsRunTask(
        synthetics_run=SyntheticsRun(
            identifier="generate",
            record_handlers={},
            lost_contact=[],
            preserved=preserved or [],
            record_size_ratio=1.0,
        ),
        synthetics_train=SyntheticsTrain(
            models={
                table: Mock(
                    create_record_handler=Mock(),
                    status=_status_for_table(table, failed or []),
                )
                for table in rel_data.list_all_tables()
                if table not in (omitted or [])
            },
        ),
        output_handler=output_handler,
        subdir="run-identifier",
        ctx=context,
        strategy=MockStrategy(),
        rel_data=rel_data,
    )


def test_ignores_preserved_tables(pets, output_handler):
    task = make_task(pets, output_handler, preserved=["pets"])

    # Source data is used
    assert task.working_tables["pets"] is not None
    assert "pets" in task.output_tables
    task.each_iteration()
    assert "pets" not in task.synthetics_run.record_handlers


def test_ignores_tables_that_were_omitted_from_training(pets, output_handler):
    task = make_task(pets, output_handler, omitted=["pets"])

    # Source data is used
    assert task.working_tables["pets"] is not None
    assert "pets" in task.output_tables
    task.each_iteration()
    assert "pets" not in task.synthetics_run.record_handlers


def test_ignores_tables_that_failed_during_training(pets, output_handler):
    task = make_task(pets, output_handler, failed=["pets"])

    # We set tables that failed to explicit None
    assert task.working_tables["pets"] is None
    assert "pets" not in task.output_tables
    task.each_iteration()
    assert "pets" not in task.synthetics_run.record_handlers


def test_runs_post_processing_when_table_completes(pets, output_handler):
    task = make_task(pets, output_handler)

    raw_df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})

    with patch(
        "gretel_trainer.relational.sdk_extras.ExtendedGretelSDK.get_record_handler_data"
    ) as get_rh_data:
        get_rh_data.return_value = raw_df
        task.handle_completed("table", Mock(ref_data=Mock(values=[])))

    post_processed = task.working_tables["table"]
    assert post_processed is not None
    pdtest.assert_frame_equal(post_processed, raw_df.head(1))


def test_starts_jobs_for_ready_tables(pets, output_handler):
    task = make_task(pets, output_handler)

    assert len(task.synthetics_run.record_handlers) == 0

    task.each_iteration()

    assert len(task.synthetics_run.record_handlers) == 1
    assert "humans" in task.synthetics_run.record_handlers
    task.synthetics_train.models[
        "humans"
    ].create_record_handler_obj.assert_called_once()
    task.synthetics_run.record_handlers["humans"].submit.assert_called_once()


def test_defers_job_submission_if_max_jobs(pets, output_handler):
    task = make_task(pets, output_handler)

    assert len(task.synthetics_run.record_handlers) == 0

    humans_model = task.synthetics_train.models["humans"]

    # If we already have the max number of jobs in flight...
    task.ctx.in_flight_jobs = MAX_IN_FLIGHT_JOBS

    task.each_iteration()

    # ...the record handler is created, but not submitted
    assert len(task.synthetics_run.record_handlers) == 1
    assert "humans" in task.synthetics_run.record_handlers
    humans_model.create_record_handler_obj.assert_called_once()
    humans_record_handler = task.synthetics_run.record_handlers["humans"]
    humans_record_handler.submit.assert_not_called()

    # Subsequent passes through the task loop will neither submit the job,
    # nor recreate a new record handler instance.
    humans_model.reset_mock()
    task.ctx.maybe_start_job(
        job=humans_record_handler,
        table_name="humans",
        action=task.action(humans_record_handler),
    )
    task.each_iteration()
    humans_model.create_record_handler_obj.assert_not_called()
    humans_record_handler.submit.assert_not_called()

    # Once there is room again for more jobs...
    task.ctx.in_flight_jobs = 0

    # ...the next pass through submits the record handler since there is now room for another job.
    task.ctx.maybe_start_job(
        job=humans_record_handler,
        table_name="humans",
        action=task.action(humans_record_handler),
    )
    humans_record_handler.submit.assert_called_once()
    assert task.ctx.in_flight_jobs == 1
