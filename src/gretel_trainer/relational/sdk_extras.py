from pathlib import Path
from typing import Dict

from gretel_client.projects import Project
from gretel_client.projects.jobs import Job, Status


def upload_singleton_project_artifact(project: Project, path: Path) -> str:
    latest_key = project.upload_artifact(str(path))
    for artifact in project.artifacts:
        key = artifact["key"]
        if key != latest_key and key.endswith(path.name):
            project.delete_artifact(key)
    return latest_key


def cautiously_refresh_status(
    job: Job, key: str, refresh_attempts: Dict[str, int]
) -> Status:
    try:
        job.refresh()
        refresh_attempts[key] = 0
    except:
        refresh_attempts[key] = refresh_attempts[key] + 1

    return job.status
