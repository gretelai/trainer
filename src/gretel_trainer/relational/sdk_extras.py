from pathlib import Path

from gretel_client.projects import Project


def upload_gretel_singleton_object(project: Project, path: Path) -> None:
    latest_key = project.upload_artifact(str(path))
    for artifact in project.artifacts:
        key = artifact["key"]
        if key != latest_key and key.endswith(path.name):
            project.delete_artifact(key)
