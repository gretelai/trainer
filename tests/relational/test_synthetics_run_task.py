import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Union
from unittest.mock import Mock, patch

import pandas as pd
import pandas.testing as pdtest
import pytest
from gretel_client.projects.projects import Project

from gretel_trainer.relational.core import RelationalData
from gretel_trainer.relational.strategies.ancestral import AncestralStrategy
from gretel_trainer.relational.strategies.independent import IndependentStrategy
from gretel_trainer.relational.tasks import SyntheticsRunTask


@dataclass
class MockMultiTable:
    relational_data: RelationalData
    _refresh_interval: int = 0
    _project: Project = Mock()
    _strategy: Union[AncestralStrategy, IndependentStrategy] = AncestralStrategy()

    def _backup(self) -> None:
        pass


@pytest.fixture(autouse=True)
def tmpdir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def mock_multitable(rel_data: RelationalData) -> MockMultiTable:
    return MockMultiTable(relational_data=rel_data)


def test_it(pets, tmpdir):
    multitable = mock_multitable(pets)
    task = SyntheticsRunTask(
        record_handlers={},
        lost_contact=[],
        record_size_ratio=1.0,
        training_columns={},
        models={},
        run_dir=tmpdir,
        working_tables={},
        multitable=multitable,
    )


def test_runs_post_processing_on_completed_record_handlers(pets, tmpdir):
    multitable = mock_multitable(pets)
    task = SyntheticsRunTask(
        record_handlers={},
        lost_contact=[],
        record_size_ratio=1.0,
        training_columns={},
        models={},
        run_dir=tmpdir,
        working_tables={},
        multitable=multitable,
    )
    raw_df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})

    class MockStrategy:
        def post_process_individual_synthetic_result(self, table, rel_data, rh_data):
            return rh_data.head(1)

    multitable._strategy = MockStrategy()  # type:ignore

    with patch(
        "gretel_trainer.relational.tasks.synthetics_run.get_record_handler_data"
    ) as get_rh_data:
        get_rh_data.return_value = raw_df
        task.handle_completed("table", Mock())

    pdtest.assert_frame_equal(task.working_tables["table"], raw_df.head(1))


def test_starts_jobs_for_ready_tables():
    pass


def test_defers_jobs_if_no_room():
    pass


def test_does_not_restart_existing_deferred_jobs():
    pass
