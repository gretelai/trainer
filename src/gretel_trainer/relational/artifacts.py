import shutil
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


def archive_items(targz: Path, items: list[Path]) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        for item in items:
            shutil.copy(item, tmpdir)
        _archive_dir(targz, Path(tmpdir))


def archive_nested_dir(targz: Path, directory: Path, name: str) -> None:
    """
    Creates an archive of the provided `directory` with name `{name}.tar.gz`
    and adds it to the provided `targz` archive (or creates it if `targz` does not yet exist).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        nested_archive = Path(f"{tmpdir}/{name}.tar.gz")
        _archive_dir(nested_archive, directory)
        _add_to_archive(targz, nested_archive)


def _archive_dir(targz: Path, directory: Path) -> None:
    shutil.make_archive(
        base_name=_base_name(targz),
        format="gztar",
        root_dir=directory,
    )


def _add_to_archive(targz: Path, item: Path) -> None:
    if targz.exists():
        with tempfile.TemporaryDirectory() as tmpdir:
            shutil.unpack_archive(targz, extract_dir=tmpdir, format="gztar")
            shutil.copy(item, tmpdir)
            _archive_dir(targz, Path(tmpdir))
    else:
        archive_items(targz, [item])


def _base_name(targz: Path) -> str:
    # shutil.make_archive base_name expects a name *without* a format-specific extension
    return str(targz).removesuffix(".tar.gz")
