import logging
from pathlib import Path
from typing import Dict, Union

import requests
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


def download_one_artifact(
    gretel_object: Union[Project, Model],
    artifact_name: str,
    out_path: Union[str, Path],
) -> None:
    download_link = gretel_object.get_artifact_link(artifact_name)
    try:
        artifact = requests.get(download_link)
        if artifact.status_code == 200:
            with open(out_path, "wb+") as out:
                out.write(artifact.content)
    except requests.exceptions.HTTPError as ex:
        logger.warning(f"Failed to download `{artifact_name}`")
