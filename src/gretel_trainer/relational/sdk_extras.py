import logging
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, Optional, Union

import requests
import smart_open
from gretel_client.projects.jobs import Job, Status
from gretel_client.projects.models import Model
from gretel_client.projects.projects import Project

logger = logging.getLogger(__name__)


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
