import time
from typing import Tuple

from gretel_client.projects.jobs import ACTIVE_STATES, END_STATES, Job, Status

from gretel_trainer.b2.core import RunIdentifier, log


def await_job(
    run_identifier: RunIdentifier, job: Job, task: str, wait: int
) -> Status:
    failed_refresh_attempts = 0
    status = job.status
    while not _finished(status):
        if failed_refresh_attempts >= 5:
            return Status.LOST

        time.sleep(wait)

        status, failed_refresh_attempts = _cautiously_refresh_status(
            job, failed_refresh_attempts
        )
        _log_in_progress(run_identifier, status, task)
    return status


def _finished(status: Status) -> bool:
    return status in END_STATES


def _cautiously_refresh_status(job: Job, attempts: int) -> Tuple[Status, int]:
    try:
        job.refresh()
        attempts = 0
    except:
        attempts = attempts + 1

    return (job.status, attempts)


def _log_in_progress(run_identifier: RunIdentifier, status: Status, task: str) -> None:
    if status in ACTIVE_STATES:
        log(run_identifier, f"{task} job still in progress (status: {status})")
