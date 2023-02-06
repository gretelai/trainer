from dataclasses import dataclass, field
from typing import Dict, Optional

from gretel_client.projects import Project


@dataclass
class ArtifactCollection:
    project: Project
    gretel_backup: Optional[str] = None
    gretel_debug_summary: Optional[str] = None
    transforms_source_tables: Dict[str, str] = field(default_factory=dict)
    transforms_output_archive: Optional[str] = None
    synthetics_source_archive: Optional[str] = None
    synthetics_train_tables: Dict[str, str] = field(default_factory=dict)
    synthetics_seeds: Dict[str, str] = field(default_factory=dict)
    synthetics_output_archive: Optional[str] = None
    synthetics_reports_archive: Optional[str] = None

    def upload_gretel_backup(self, path: str) -> None:
        existing = self.gretel_backup
        self.gretel_backup = self._upload_file(path, existing)

    def upload_gretel_debug_summary(self, path: str) -> None:
        existing = self.gretel_debug_summary
        self.gretel_debug_summary = self._upload_file(path, existing)

    def upload_transforms_source_table(self, path: str, table_name: str) -> None:
        existing = self.transforms_source_tables.get(table_name)
        self.transforms_source_tables[table_name] = self._upload_file(path, existing)

    def upload_transforms_output_archive(self, path: str) -> None:
        existing = self.transforms_output_archive
        self.transforms_output_archive = self._upload_file(path, existing)

    def upload_synthetics_source_archive(self, path: str) -> None:
        existing = self.synthetics_source_archive
        self.synthetics_source_archive = self._upload_file(path, existing)

    def upload_synthetics_train_table(self, path: str, table_name: str) -> None:
        existing = self.synthetics_train_tables.get(table_name)
        self.synthetics_train_tables[table_name] = self._upload_file(path, existing)

    def upload_synthetics_seed(self, path: str, table_name: str) -> None:
        existing = self.synthetics_seeds.get(table_name)
        self.synthetics_seeds[table_name] = self._upload_file(path, existing)

    def upload_synthetics_output_archive(self, path: str) -> None:
        existing = self.synthetics_output_archive
        self.synthetics_output_archive = self._upload_file(path, existing)

    def upload_synthetics_reports_archive(self, path: str) -> None:
        existing = self.synthetics_reports_archive
        self.synthetics_reports_archive = self._upload_file(path, existing)

    def _upload_file(self, path: str, existing: Optional[str]) -> str:
        latest = self.project.upload_artifact(path)
        if existing is not None:
            self.project.delete_artifact(existing)
        return latest