import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
from unittest.mock import Mock, patch

import pandas as pd
import pandas.testing as pdtest
import pytest
from gretel_client.projects.jobs import Status
from gretel_client.projects.projects import Project

from gretel_trainer.relational.core import RelationalData
from gretel_trainer.relational.sdk_extras import (
    MAX_PROJECT_ARTIFACTS,
    ExtendedGretelSDK,
)
from gretel_trainer.relational.strategies.ancestral import AncestralStrategy
from gretel_trainer.relational.strategies.independent import IndependentStrategy
from gretel_trainer.relational.tasks import SyntheticsRunTask
from gretel_trainer.relational.workflow_state import SyntheticsRun, SyntheticsTrain


@dataclass
class MockMultiTable:
    relational_data: RelationalData
    _refresh_interval: int = 0
    _project: Project = Mock(artifacts=[])
    _strategy: Union[AncestralStrategy, IndependentStrategy] = AncestralStrategy()
    _extended_sdk: ExtendedGretelSDK = ExtendedGretelSDK(hybrid=False)

    def _backup(self) -> None:
        pass


@pytest.fixture(autouse=True)
def tmpdir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def make_task(
    rel_data: RelationalData,
    run_dir: Path,
    preserved: Optional[list[str]] = None,
    failed: Optional[list[str]] = None,
    omitted: Optional[list[str]] = None,
) -> SyntheticsRunTask:
    def _status_for_table(table: str, failed: list[str]) -> Status:
        if table in failed:
            return Status.ERROR
        else:
            return Status.COMPLETED

    multitable = MockMultiTable(relational_data=rel_data)
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
        run_dir=run_dir,
        multitable=multitable,
    )


def test_ignores_preserved_tables(pets, tmpdir):
    task = make_task(pets, tmpdir, preserved=["pets"])

    # Source data is used
    assert task.working_tables["pets"] is not None
    assert "pets" in task.output_tables
    task.each_iteration()
    assert "pets" not in task.synthetics_run.record_handlers


def test_ignores_tables_that_were_omitted_from_training(pets, tmpdir):
    task = make_task(pets, tmpdir, omitted=["pets"])

    # Source data is used
    assert task.working_tables["pets"] is not None
    assert "pets" in task.output_tables
    task.each_iteration()
    assert "pets" not in task.synthetics_run.record_handlers


def test_ignores_tables_that_failed_during_training(pets, tmpdir):
    task = make_task(pets, tmpdir, failed=["pets"])

    # We set tables that failed to explicit None
    assert task.working_tables["pets"] is None
    assert "pets" not in task.output_tables
    task.each_iteration()
    assert "pets" not in task.synthetics_run.record_handlers


def test_runs_post_processing_when_table_completes(pets, tmpdir):
    task = make_task(pets, tmpdir)

    raw_df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})

    class MockStrategy:
        def post_process_individual_synthetic_result(
            self, table_name, rel_data, synthetic_table, record_size_ratio
        ):
            return synthetic_table.head(1)

    task.multitable._strategy = MockStrategy()  # type:ignore

    with patch(
        "gretel_trainer.relational.sdk_extras.ExtendedGretelSDK.get_record_handler_data"
    ) as get_rh_data:
        get_rh_data.return_value = raw_df
        task.handle_completed("table", Mock())

    post_processed = task.working_tables["table"]
    assert post_processed is not None
    pdtest.assert_frame_equal(post_processed, raw_df.head(1))


def test_starts_jobs_for_ready_tables(pets, tmpdir):
    task = make_task(pets, tmpdir)

    assert len(task.synthetics_run.record_handlers) == 0

    task.each_iteration()

    assert len(task.synthetics_run.record_handlers) == 1
    assert "humans" in task.synthetics_run.record_handlers
    task.synthetics_train.models[
        "humans"
    ].create_record_handler_obj.assert_called_once()
    task.synthetics_run.record_handlers["humans"].submit.assert_called_once()


def test_defers_jobs_if_no_room(pets, tmpdir):
    task = make_task(pets, tmpdir)
    task.multitable._project.artifacts = ["art"] * MAX_PROJECT_ARTIFACTS

    assert len(task.synthetics_run.record_handlers) == 0

    task.each_iteration()

    # We create the record handler, but defer submitting it because the project has no room
    # Note: the `humans` table is a parent table and ordinarily does not have a data source,
    # and therefore wouldn't be deferred; in this context, though, the record handler object
    # is a Mock and calling the `data_source` property returns another mock object, not None
    assert len(task.synthetics_run.record_handlers) == 1
    assert "humans" in task.synthetics_run.record_handlers
    task.synthetics_train.models[
        "humans"
    ].create_record_handler_obj.assert_called_once()
    task.synthetics_run.record_handlers["humans"].submit.assert_not_called()


def test_does_not_restart_existing_deferred_jobs(pets, tmpdir):
    task = make_task(pets, tmpdir)
    task.multitable._project.artifacts = ["art"] * MAX_PROJECT_ARTIFACTS

    assert len(task.synthetics_run.record_handlers) == 0

    task.each_iteration()

    # We create the record handler, but defer submitting it
    assert len(task.synthetics_run.record_handlers) == 1
    assert "humans" in task.synthetics_run.record_handlers
    task.synthetics_train.models[
        "humans"
    ].create_record_handler_obj.assert_called_once()
    task.synthetics_run.record_handlers["humans"].submit.assert_not_called()

    task.synthetics_train.models["humans"].reset_mock()

    # On next iteration, we do not *re-create* a record handler object for humans
    task.each_iteration()
    task.synthetics_train.models["humans"].create_record_handler_obj.assert_not_called()
