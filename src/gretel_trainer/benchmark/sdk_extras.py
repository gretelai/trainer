import json
import time
from typing import Any

import smart_open
from gretel_client.projects.jobs import (
    ACTIVE_STATES,
    END_STATES,
    Job,
    RunnerMode,
    Status,
)
from gretel_client.projects.models import Model, read_model_config
from gretel_client.projects.projects import Project

from gretel_trainer.benchmark.core import BenchmarkException, log


def create_evaluate_model(
    project: Project,
    data_source: str,
    ref_data: str,
    run_identifier: str,
) -> Model:
    return project.create_model_obj(
        model_config=_make_evaluate_config(run_identifier),
        data_source=data_source,
        ref_data=ref_data,
    )


def run_evaluate(
    evaluate_model: Model,
    run_identifier: str,
    wait: int,
) -> dict[str, Any]:
    # Calling this in lieu of submit_cloud() is supposed to avoid
    # artifact upload. Doesn't work for more recent client versions!
    evaluate_model.submit(runner_mode=RunnerMode.CLOUD)
    job_status = await_job(run_identifier, evaluate_model, "evaluation", wait)
    if job_status in END_STATES and job_status != Status.COMPLETED:
        raise BenchmarkException("Evaluate failed")
    return json.loads(
        smart_open.open(evaluate_model.get_artifact_link("report_json")).read()
    )


def _make_evaluate_config(run_identifier: str) -> dict:
    config_dict = read_model_config("evaluate/default")
    config_dict["name"] = f"evaluate-{run_identifier}"
    return config_dict


def await_job(run_identifier: str, job: Job, task: str, wait: int) -> Status:
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


def _cautiously_refresh_status(job: Job, attempts: int) -> tuple[Status, int]:
    try:
        job.refresh()
        attempts = 0
    except:
        attempts = attempts + 1

    return (job.status, attempts)


def _log_in_progress(run_identifier: str, status: Status, task: str) -> None:
    if status in ACTIVE_STATES:
        log(run_identifier, f"{task} job still in progress (status: {status})")
