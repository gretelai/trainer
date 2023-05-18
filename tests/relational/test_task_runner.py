from unittest.mock import Mock, PropertyMock, patch

import pytest
from gretel_client.projects.jobs import Job, Status

from gretel_trainer.relational.sdk_extras import MAX_PROJECT_ARTIFACTS
from gretel_trainer.relational.task_runner import run_task


class MockTask:
    def __init__(self, project, models):
        self.project = project
        self.models = models
        self.iteration_count = 0
        self.completed = []
        self.failed = []
        self.lost_contact = []

    def action(self, job: Job) -> str:
        return "mock task"

    @property
    def artifacts_per_job(self) -> int:
        return 3

    @property
    def multitable(self):
        return Mock()

    @property
    def table_collection(self) -> list[str]:
        return list(self.models.keys())

    def more_to_do(self) -> bool:
        return len(self.completed + self.failed + self.lost_contact) < len(self.models)

    def wait(self) -> None:
        pass

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


@pytest.fixture(autouse=True)
def get_job_id():
    with patch(
        "gretel_trainer.relational.sdk_extras.ExtendedGretelSDK.get_job_id"
    ) as _get_job_id:
        yield _get_job_id


def test_one_successful_model(get_job_id, extended_sdk):
    get_job_id.side_effect = [None, "id"]

    project = Mock(artifacts=[])
    model = Mock(status=Status.COMPLETED)
    models = {"table": model}

    task = MockTask(
        project=project,
        models=models,
    )
    run_task(task, extended_sdk)

    assert task.iteration_count == 2
    assert task.completed == ["table"]
    assert task.failed == []


def test_one_failed_model(get_job_id, extended_sdk):
    get_job_id.side_effect = [None, "id"]

    project = Mock(artifacts=[])
    model = Mock(status=Status.ERROR)
    models = {"table": model}

    task = MockTask(
        project=project,
        models=models,
    )
    run_task(task, extended_sdk)

    assert task.iteration_count == 2
    assert task.completed == []
    assert task.failed == ["table"]


def test_model_taking_awhile(get_job_id, extended_sdk):
    get_job_id.side_effect = [None, "id", "id"]

    project = Mock(artifacts=[])
    model = Mock()
    status = PropertyMock(side_effect=[Status.ACTIVE, Status.COMPLETED])
    type(model).status = status
    models = {"table": model}

    task = MockTask(
        project=project,
        models=models,
    )
    run_task(task, extended_sdk)

    assert task.iteration_count == 3
    assert task.completed == ["table"]
    assert task.failed == []


def test_lose_contact_with_model(get_job_id, extended_sdk):
    get_job_id.side_effect = [None, "id", "id", "id"]

    project = Mock(artifacts=[])
    model = Mock(status=Status.ACTIVE)
    model.refresh.side_effect = Exception()
    models = {"table": model}

    task = MockTask(
        project=project,
        models=models,
    )
    run_task(task, extended_sdk)

    # Bail after refresh fails MAX_REFRESH_ATTEMPTS times
    assert task.iteration_count == 4
    assert task.completed == []
    assert task.failed == []
    assert task.lost_contact == ["table"]


def test_refresh_status_can_tolerate_blips(get_job_id, extended_sdk):
    get_job_id.side_effect = [None, "id", "id"]

    project = Mock(artifacts=[])
    model = Mock()
    status = PropertyMock(side_effect=[Status.ACTIVE, Status.COMPLETED])
    type(model).status = status
    model.refresh.side_effect = [Exception(), None]
    models = {"table": model}

    task = MockTask(
        project=project,
        models=models,
    )
    run_task(task, extended_sdk)

    assert task.iteration_count == 3
    assert task.completed == ["table"]
    assert task.failed == []
    assert task.lost_contact == []


def test_defers_submission_if_no_room_in_project(get_job_id, extended_sdk):
    get_job_id.side_effect = [None, None, None, "id"]
    project = Mock()
    # First time through, we're at the project limit
    # Second time through, we're below the limit, but still not enough room for this task
    # Third time through, there is enough space
    artifacts = PropertyMock(
        side_effect=[
            ["art"] * MAX_PROJECT_ARTIFACTS,
            ["art"] * (MAX_PROJECT_ARTIFACTS - 1),
            ["art"] * (MAX_PROJECT_ARTIFACTS - 3),
        ]
    )
    type(project).artifacts = artifacts
    model = Mock(status=Status.COMPLETED)
    models = {"table": model}

    task = MockTask(
        project=project,
        models=models,
    )
    run_task(task, extended_sdk)

    assert task.iteration_count == 4
    assert task.completed == ["table"]


def test_several_models(extended_sdk):
    project = Mock(artifacts=[])

    completed_model = Mock(status=Status.COMPLETED)
    error_model = Mock(status=Status.ERROR)
    cancelled_model = Mock(status=Status.CANCELLED)
    lost_model = Mock(status=Status.LOST)

    models = {
        "completed": completed_model,
        "error": error_model,
        "cancelled": cancelled_model,
        "lost": lost_model,
    }

    task = MockTask(
        project=project,
        models=models,
    )
    run_task(task, extended_sdk)

    assert task.completed == ["completed"]
    assert set(task.failed) == {"error", "cancelled", "lost"}
