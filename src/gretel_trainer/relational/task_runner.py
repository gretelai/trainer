import logging
from collections import defaultdict
from typing import Protocol

from gretel_client.projects.jobs import END_STATES, Job, Status
from gretel_client.projects.projects import Project

from gretel_trainer.relational.sdk_extras import ExtendedGretelSDK

MAX_REFRESH_ATTEMPTS = 3

logger = logging.getLogger(__name__)


class Task(Protocol):
    def action(self, job: Job) -> str:
        ...

    @property
    def table_collection(self) -> list[str]:
        ...

    @property
    def artifacts_per_job(self) -> int:
        ...

    @property
    def project(self) -> Project:
        ...

    def more_to_do(self) -> bool:
        ...

    def wait(self) -> None:
        ...

    def is_finished(self, table: str) -> bool:
        ...

    def get_job(self, table: str) -> Job:
        ...

    def handle_completed(self, table: str, job: Job) -> None:
        ...

    def handle_failed(self, table: str, job: Job) -> None:
        ...

    def handle_in_progress(self, table: str, job: Job) -> None:
        ...

    def handle_lost_contact(self, table: str, job: Job) -> None:
        ...

    def each_iteration(self) -> None:
        ...


def run_task(task: Task, extended_sdk: ExtendedGretelSDK) -> None:
    refresh_attempts: dict[str, int] = defaultdict(int)
    first_pass = True

    while task.more_to_do():
        if first_pass:
            first_pass = False
        else:
            task.wait()

        for table_name in task.table_collection:
            if task.is_finished(table_name):
                continue

            job = task.get_job(table_name)
            if extended_sdk.get_job_id(job) is None:
                extended_sdk.start_job_if_possible(
                    job=job,
                    table_name=table_name,
                    action=task.action(job),
                    project=task.project,
                    number_of_artifacts=task.artifacts_per_job,
                )
                continue

            status = extended_sdk.cautiously_refresh_status(
                job, table_name, refresh_attempts
            )

            if refresh_attempts[table_name] >= MAX_REFRESH_ATTEMPTS:
                task.handle_lost_contact(table_name, job)
                continue

            if status == Status.COMPLETED:
                task.handle_completed(table_name, job)
            elif status in END_STATES:
                task.handle_failed(table_name, job)
            else:
                task.handle_in_progress(table_name, job)

        task.each_iteration()
