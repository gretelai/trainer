from dataclasses import dataclass
from typing import Optional

from gretel_client.projects import Project


@dataclass
class ArtifactCollection:
    gretel_debug_summary: Optional[str] = None
    transforms_output_archive: Optional[str] = None
    synthetics_source_archive: Optional[str] = None
    synthetics_output_archive: Optional[str] = None
    synthetics_reports_archive: Optional[str] = None

    # Backup behaves differently to avoid constantly busting the cache
    def upload_gretel_backup(self, project: Project, path: str) -> None:
        latest = project.upload_artifact(path)
        for artifact in project.artifacts:
            key = artifact["key"]
            if key != latest and key.endswith("__gretel_backup.json"):
                project.delete_artifact(key)

    def upload_gretel_debug_summary(self, project: Project, path: str) -> None:
        existing = self.gretel_debug_summary
        self.gretel_debug_summary = self._upload_file(project, path, existing)

    def upload_transforms_output_archive(self, project: Project, path: str) -> None:
        existing = self.transforms_output_archive
        self.transforms_output_archive = self._upload_file(project, path, existing)

    def upload_synthetics_source_archive(self, project: Project, path: str) -> None:
        existing = self.synthetics_source_archive
        self.synthetics_source_archive = self._upload_file(project, path, existing)

    def upload_synthetics_output_archive(self, project: Project, path: str) -> None:
        existing = self.synthetics_output_archive
        self.synthetics_output_archive = self._upload_file(project, path, existing)

    def upload_synthetics_reports_archive(self, project: Project, path: str) -> None:
        existing = self.synthetics_reports_archive
        self.synthetics_reports_archive = self._upload_file(project, path, existing)

    def _upload_file(self, project: Project, path: str, existing: Optional[str]) -> str:
        latest = project.upload_artifact(path)
        if existing is not None:
            project.delete_artifact(existing)
        return latest
