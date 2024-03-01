from typing import Optional
from unittest.mock import Mock, patch

import pytest

from gretel_client.projects.exceptions import MaxConcurrentJobsException
from gretel_client.projects.jobs import Job, Status
from gretel_trainer.relational.sdk_extras import ExtendedGretelSDK
from gretel_trainer.relational.task_runner import run_task, TaskContext


class MockTask:
    def __init__(self, project, models):
        self.project = project
        self.models = models
        self.iteration_count = 0
        self.completed = []
        self.failed = []
        self.lost_contact = []
        self.ctx = TaskContext(
            in_flight_jobs=0,
            refresh_interval=0,
            project=project,
            extended_sdk=ExtendedGretelSDK(hybrid=False),
            backup=lambda: None,
        )

    def action(self, job: Job) -> str:
        return "mock task"

    @property
    def table_collection(self) -> list[str]:
        return list(self.models.keys())

    def more_to_do(self) -> bool:
        return len(self.completed + self.failed + self.lost_contact) < len(self.models)

    def is_finished(self, table: str) -> bool:
        return table in (self.completed + self.failed + self.lost_contact)

    def get_job(self, table: str) -> Job:
        return self.models[table]

    def handle_completed(self, table: str, job: Job) -> None:
        self.completed.append(table)

    def handle_failed(self, table: str, job: Job) -> None:
        self.failed.append(table)

    def handle_lost_contact(self, table: str, job: Job) -> None:
        self.lost_contact.append(table)

    def handle_in_progress(self, table: str, job: Job) -> None:
        pass

    def each_iteration(self) -> None:
        self.iteration_count += 1


class MockModel:
    def __init__(self, statuses: list[Optional[str]], fail_n_times: int = 0):
        self.identifier = None

        self._statuses = statuses
        self.status = None

        self._fail_n_times = fail_n_times
        self._fail_count = 0

    def submit(self):
        if self._fail_count < self._fail_n_times:
            self._fail_count += 1
            raise MaxConcurrentJobsException()
        self.identifier = "identifier"

    def refresh(self):
        next_status = self._statuses.pop(0)
        if next_status is None:
            raise Exception()
        self.status = next_status


@pytest.fixture(autouse=True)
def mock_extended_sdk():
    def _get_job_id(mock_model):
        return mock_model.identifier

    extended_sdk = ExtendedGretelSDK(hybrid=False)
    extended_sdk.get_job_id = _get_job_id  # type:ignore
    return extended_sdk


def test_one_successful_model(mock_extended_sdk):
    models = {
        "table": MockModel(statuses=[Status.COMPLETED]),
    }

    task = MockTask(
        project=Mock(),
        models=models,
    )
    run_task(task, mock_extended_sdk)

    assert task.iteration_count == 2
    assert task.completed == ["table"]
    assert task.failed == []


def test_one_failed_model(mock_extended_sdk):
    models = {
        "table": MockModel(statuses=[Status.ERROR]),
    }

    task = MockTask(
        project=Mock(),
        models=models,
    )
    run_task(task, mock_extended_sdk)

    assert task.iteration_count == 2
    assert task.completed == []
    assert task.failed == ["table"]


def test_model_taking_awhile(mock_extended_sdk):
    models = {
        "table": MockModel(statuses=[Status.ACTIVE, Status.ACTIVE, Status.COMPLETED]),
    }

    task = MockTask(
        project=Mock(),
        models=models,
    )
    run_task(task, mock_extended_sdk)

    assert task.iteration_count == 4
    assert task.completed == ["table"]
    assert task.failed == []


def test_lose_contact_with_model(mock_extended_sdk):
    # By only setting one status, subsequent calls to `refresh` will throw
    # an IndexError (as a stand-in for SDK refresh errors)
    models = {
        "table": MockModel(statuses=[Status.ACTIVE]),
    }

    task = MockTask(
        project=Mock(),
        models=models,
    )
    run_task(task, mock_extended_sdk)

    # Bail after refresh fails MAX_REFRESH_ATTEMPTS times
    # (first iteration creates the job, +4 refresh failures)
    assert task.iteration_count == 5
    assert task.completed == []
    assert task.failed == []
    assert task.lost_contact == ["table"]


def test_refresh_status_can_tolerate_blips(mock_extended_sdk):
    models = {
        "table": MockModel(
            statuses=[Status.ACTIVE, None, Status.ACTIVE, Status.COMPLETED]
        ),
    }

    task = MockTask(
        project=Mock(),
        models=models,
    )
    run_task(task, mock_extended_sdk)

    # 1. Create
    # 2. Active
    # 3. Blip
    # 4. Active
    # 5. Completed
    assert task.iteration_count == 5
    assert task.completed == ["table"]
    assert task.failed == []
    assert task.lost_contact == []


def test_defers_submission_if_max_jobs_in_flight(mock_extended_sdk):
    model_1 = MockModel(statuses=[Status.ACTIVE, Status.COMPLETED])
    model_2 = MockModel(statuses=[Status.ACTIVE, Status.COMPLETED])

    models = {"t1": model_1, "t2": model_2}

    task = MockTask(
        project=Mock(),
        models=models,
    )
    with patch("gretel_trainer.relational.sdk_extras.MAX_IN_FLIGHT_JOBS", 1):
        run_task(task, mock_extended_sdk)

    # 1: Started, Deferred
    # 2: Active, Deferred
    # 3: Completed, Started
    # 4: Completed, Active
    # 5: Completed, Completed
    assert task.iteration_count == 5
    assert task.completed == ["t1", "t2"]


def test_defers_submission_if_max_jobs_in_created_state(mock_extended_sdk):
    # In this test, we're not running into our client-side max jobs limit;
    # rather, the API is not allowing us to submit due to too many jobs in created state.
    # The second model fails to be submitted 5 times (e.g. due to other unrelated jobs)
    # before getting submitted successfully
    model_1 = MockModel(statuses=[Status.ACTIVE, Status.COMPLETED])
    model_2 = MockModel(statuses=[Status.ACTIVE, Status.COMPLETED], fail_n_times=5)

    models = {"t1": model_1, "t2": model_2}

    task = MockTask(
        project=Mock(),
        models=models,
    )
    run_task(task, mock_extended_sdk)

    # 1: Started, Deferred
    # 2: Active, Deferred
    # 3: Completed, Deferred
    # 4: Completed, Deferred
    # 5: Completed, Deferred
    # 6: Completed, Started
    # 7: Completed, Active
    # 8: Completed, Completed
    assert task.iteration_count == 8
    assert task.completed == ["t1", "t2"]


def test_several_models(mock_extended_sdk):
    completed_model = MockModel(statuses=[Status.COMPLETED])
    error_model = MockModel(statuses=[Status.ERROR])
    cancelled_model = MockModel(statuses=[Status.CANCELLED])
    lost_model = MockModel(statuses=[Status.LOST])

    models = {
        "completed": completed_model,
        "error": error_model,
        "cancelled": cancelled_model,
        "lost": lost_model,
    }

    task = MockTask(
        project=Mock(),
        models=models,
    )
    run_task(task, mock_extended_sdk)

    assert task.completed == ["completed"]
    assert set(task.failed) == {"error", "cancelled", "lost"}
