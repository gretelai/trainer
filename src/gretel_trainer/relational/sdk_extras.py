import logging
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import requests
import smart_open
from gretel_client.projects.jobs import Job, Status
from gretel_client.projects.models import Model
from gretel_client.projects.projects import Project
from gretel_client.projects.records import RecordHandler

from gretel_trainer.relational.core import MultiTableException

logger = logging.getLogger(__name__)

MAX_PROJECT_ARTIFACTS = 50


def get_job_id(job: Job) -> Optional[str]:
    if isinstance(job, Model):
        return job.model_id
    elif isinstance(job, RecordHandler):
        return job.record_id
    else:
        raise MultiTableException("Unexpected job object")


def room_in_project(project: Project, count: int = 1) -> bool:
    return len(project.artifacts) + count <= MAX_PROJECT_ARTIFACTS


def delete_data_source(project: Project, job: Job) -> None:
    if job.data_source is not None:
        project.delete_artifact(job.data_source)


def cautiously_refresh_status(
    job: Job, key: str, refresh_attempts: Dict[str, int]
) -> Status:
    try:
        job.refresh()
        refresh_attempts[key] = 0
    except:
        refresh_attempts[key] = refresh_attempts[key] + 1

    return job.status


def download_file_artifact(
    gretel_object: Union[Project, Model],
    artifact_name: str,
    out_path: Union[str, Path],
) -> None:
    download_link = gretel_object.get_artifact_link(artifact_name)
    try:
        with open(out_path, "wb+") as out:
            out.write(smart_open.open(download_link, "rb").read())
    except:
        logger.warning(f"Failed to download `{artifact_name}`")


def download_tar_artifact(project: Project, artifact_name: str, out_path: Path) -> None:
    download_link = project.get_artifact_link(artifact_name)
    try:
        response = requests.get(download_link)
        if response.status_code == 200:
            with open(out_path, "wb") as out:
                out.write(response.content)
    except:
        logger.warning(f"Failed to download `{artifact_name}`")


def sqs_score_from_full_report(report: Dict[str, Any]) -> Optional[int]:
    with suppress(KeyError):
        for field_dict in report["summary"]:
            if field_dict["field"] == "synthetic_data_quality_score":
                return field_dict["value"]


def get_record_handler_data(record_handler: RecordHandler) -> pd.DataFrame:
    return pd.read_csv(record_handler.get_artifact_link("data"), compression="gzip")


def start_job_if_possible(
    job: Job,
    table_name: str,
    action: str,
    project: Project,
    number_of_artifacts: int,
) -> None:
    if job.data_source is None or room_in_project(project, number_of_artifacts):
        _log_start(table_name, action)
        job.submit_cloud()
    else:
        _log_waiting(table_name, action)


def _log_start(table_name: str, action: str) -> None:
    logger.info(f"Starting {action} for `{table_name}`.")


def _log_waiting(table_name: str, action: str) -> None:
    logger.info(
        f"Maximum concurrent relational jobs reached. Deferring start of `{table_name}` {action}."
    )
