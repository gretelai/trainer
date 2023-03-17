import logging
import time
from collections import defaultdict
from typing import Dict, List

from gretel_client.projects.jobs import ACTIVE_STATES, END_STATES, Job, Status
from gretel_client.projects.projects import Project
from typing_extensions import Protocol

from gretel_trainer.relational.sdk_extras import (
    cautiously_refresh_status,
    delete_data_source,
    get_job_id,
    room_in_project,
    start_job_if_possible,
)

MAX_REFRESH_ATTEMPTS = 3

logger = logging.getLogger(__name__)


class Task(Protocol):
    @property
    def action(self) -> str:
        ...

    @property
    def refresh_interval(self) -> int:
        ...

    @property
    def table_collection(self) -> List[str]:
        ...

    @property
    def artifacts_per_job(self) -> int:
        ...

    @property
    def project(self) -> Project:
        ...

    def more_to_do(self) -> bool:
        ...

    def is_finished(self, table: str) -> bool:
        ...

    def get_job(self, table: str) -> Job:
        ...

    def handle_completed(self, table: str, job: Job) -> None:
        ...

    def handle_failed(self, table: str) -> None:
        ...

    def handle_lost_contact(self, table: str) -> None:
        ...

    def each_iteration(self) -> None:
        ...


def run_task(task: Task) -> None:
    refresh_attempts: Dict[str, int] = defaultdict(int)
    first_pass = True

    while task.more_to_do():
        if first_pass:
            first_pass = False
        else:
            _wait(task.refresh_interval)

        for table_name in task.table_collection:
            if task.is_finished(table_name):
                continue

            job = task.get_job(table_name)
            if get_job_id(job) is None:
                start_job_if_possible(
                    job=job,
                    table_name=table_name,
                    action=task.action,
                    project=task.project,
                    number_of_artifacts=task.artifacts_per_job,
                )
                continue

            status = cautiously_refresh_status(job, table_name, refresh_attempts)

            if refresh_attempts[table_name] >= MAX_REFRESH_ATTEMPTS:
                _log_lost_contact(table_name)
                task.handle_lost_contact(table_name)
                delete_data_source(task.project, job)
                continue

            if status == Status.COMPLETED:
                _log_success(table_name, task.action)
                task.handle_completed(table_name, job)
                delete_data_source(task.project, job)
            elif status in END_STATES:
                _log_failed(table_name, task.action)
                task.handle_failed(table_name)
                delete_data_source(task.project, job)
            else:
                _log_in_progress(table_name, status, task.action)

        task.each_iteration()


def _wait(seconds: int) -> None:
    logger.info(f"Next status check in {seconds} seconds.")
    time.sleep(seconds)


def _log_lost_contact(table_name: str) -> None:
    logger.warning(f"Lost contact with job for `{table_name}`.")


def _log_success(table_name: str, action: str) -> None:
    logger.info(f"{action.capitalize()} successfully completed for `{table_name}`.")


def _log_failed(table_name: str, action: str) -> None:
    logger.info(f"{action.capitalize()} failed for `{table_name}`.")


def _log_in_progress(table_name: str, status: Status, action: str) -> None:
    logger.info(
        f"{action.capitalize()} job for `{table_name}` still in progress (status: {status})."
    )
