from dataclasses import dataclass
from typing import Optional

from gretel_client.projects import Project


@dataclass
class ArtifactCollection:
    gretel_debug_summary: Optional[str] = None
    source_archive: Optional[str] = None
    synthetics_outputs_archive: Optional[str] = None
    transforms_output_archive: Optional[str] = None

    def upload_gretel_debug_summary(self, project: Project, path: str) -> None:
        existing = self.gretel_debug_summary
        self.gretel_debug_summary = self._upload_file(project, path, existing)

    def upload_source_archive(self, project: Project, path: str) -> None:
        existing = self.source_archive
        self.source_archive = self._upload_file(project, path, existing)

    def upload_synthetics_outputs_archive(self, project: Project, path: str) -> None:
        existing = self.synthetics_outputs_archive
        self.synthetics_outputs_archive = self._upload_file(project, path, existing)

    def upload_transforms_output_archive(self, project: Project, path: str) -> None:
        existing = self.transforms_output_archive
        self.transforms_output_archive = self._upload_file(project, path, existing)

    def _upload_file(self, project: Project, path: str, existing: Optional[str]) -> str:
        latest = project.upload_artifact(path)
        if existing is not None:
            project.delete_artifact(existing)
        return latest
