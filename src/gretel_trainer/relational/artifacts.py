import os
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from gretel_client.projects import Project


@dataclass
class ArtifactCollection:
    hybrid: bool
    gretel_debug_summary: Optional[str] = None
    source_archive: Optional[str] = None
    classify_outputs_archive: Optional[str] = None
    synthetics_training_archive: Optional[str] = None
    synthetics_outputs_archive: Optional[str] = None
    transforms_outputs_archive: Optional[str] = None

    def upload_gretel_debug_summary(self, project: Project, path: str) -> None:
        existing = self.gretel_debug_summary
        self.gretel_debug_summary = self._upload_file(project, path, existing)

    def upload_source_archive(self, project: Project, path: str) -> None:
        existing = self.source_archive
        self.source_archive = self._upload_file(project, path, existing)

    def upload_classify_outputs_archive(self, project: Project, path: str) -> None:
        existing = self.classify_outputs_archive
        self.classify_outputs_archive = self._upload_file(project, path, existing)

    def upload_synthetics_training_archive(self, project: Project, path: str) -> None:
        existing = self.synthetics_training_archive
        self.synthetics_training_archive = self._upload_file(project, path, existing)

    def upload_synthetics_outputs_archive(self, project: Project, path: str) -> None:
        existing = self.synthetics_outputs_archive
        self.synthetics_outputs_archive = self._upload_file(project, path, existing)

    def upload_transforms_outputs_archive(self, project: Project, path: str) -> None:
        existing = self.transforms_outputs_archive
        self.transforms_outputs_archive = self._upload_file(project, path, existing)

    def _upload_file(
        self, project: Project, path: str, existing: Optional[str]
    ) -> Optional[str]:
        # We do not upload any of these artifacts in hybrid contexts because they are intended to be
        # "singleton" objects, but we cannot list or delete items in users' artifact endpoints, so
        # we would end up polluting their endpoints with many nearly-duplicative copies of these files.
        if self.hybrid:
            return None

        latest = project.upload_artifact(path)
        if existing is not None:
            project.delete_artifact(existing)
        return latest


def add_to_tar(targz: Path, src: Path, arcname: str) -> None:
    if targz.exists():
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            backup = tmpdir / "backup.tar.gz"
            shutil.copy(targz, backup)

            with tarfile.open(targz, "w:gz") as w, tarfile.open(backup, "r:gz") as r:
                w.add(src, arcname=arcname)

                r.extractall(tmpdir)
                for member in r.getnames():
                    if os.path.isfile(tmpdir / member) and not member == arcname:
                        w.add(tmpdir / member, arcname=member)
    else:
        with tarfile.open(targz, "w:gz") as tar:
            tar.add(src, arcname=arcname)
