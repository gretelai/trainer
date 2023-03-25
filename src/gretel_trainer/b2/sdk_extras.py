import json
import time
from typing import Any, Dict, Tuple

import smart_open
from gretel_client.projects.jobs import ACTIVE_STATES, END_STATES, Job, Status
from gretel_client.projects.models import read_model_config
from gretel_client.projects.projects import Project

from gretel_trainer.b2.core import BenchmarkException, log


def run_evaluate(
    project: Project,
    data_source: str,
    ref_data: str,
    run_identifier: str,
    wait: int,
) -> Dict[str, Any]:
    evaluate_model = project.create_model_obj(
        model_config=_make_evaluate_config(run_identifier),
        data_source=data_source,
        ref_data=ref_data,
    )
    evaluate_model.submit_cloud()
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


def _cautiously_refresh_status(job: Job, attempts: int) -> Tuple[Status, int]:
    try:
        job.refresh()
        attempts = 0
    except:
        attempts = attempts + 1

    return (job.status, attempts)


def _log_in_progress(run_identifier: str, status: Status, task: str) -> None:
    if status in ACTIVE_STATES:
        log(run_identifier, f"{task} job still in progress (status: {status})")
