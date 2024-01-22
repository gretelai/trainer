import json
import logging
import shutil
import tempfile

from pathlib import Path
from typing import Optional, Protocol

from gretel_client.projects import Project
from gretel_trainer.relational.backup import Backup
from gretel_trainer.relational.core import RelationalData

logger = logging.getLogger(__name__)


class OutputHandler(Protocol):
    def filepath_for(self, filename: str, subdir: Optional[str] = None) -> str:
        """
        Returns a string handle that can be used with smart_open to write data to an internal location.
        """
        ...

    def make_subdirectory(self, name: str) -> str:
        """
        Returns a string that can be passed to `filepath_for` to support output organization.
        """
        ...

    def get_source_archive(self) -> Optional[str]:
        """
        Returns an ID for a source archive artifact if one exists.
        """

    def save_sources(self, relational_data: RelationalData) -> None:
        """
        Callback when source data is finalized for an implementation to persist source data.
        """
        ...

    def save_backup(self, backup: Backup) -> None:
        """
        Callback at several notable moments of execution for an implementation to persist backup data.
        """
        ...

    def save_debug_summary(self, content: dict) -> None:
        """
        Callback when initial state is set up to persist debug information.
        """
        ...

    def save_classify_outputs(self, filepaths: dict[str, str]) -> None:
        """
        Callback when classify completes to persist classify output data.
        """
        ...

    def save_transforms_outputs(
        self, filepaths: dict[str, str], run_subdir: str
    ) -> None:
        """
        Callback when transform completes to persist transform output data.
        """
        ...

    def save_synthetics_outputs(
        self,
        tables: dict[str, str],
        table_reports: dict[str, dict[str, dict[str, str]]],
        relational_report: Optional[str],
        run_subdir: str,
    ) -> None:
        """
        Callback when synthetics completes to persist synthetic output data.
        """
        ...


class SDKOutputHandler:
    def __init__(
        self,
        workdir: str,
        project: Project,
        hybrid: bool,
        source_archive: Optional[str],
    ):
        self._project = project
        self._hybrid = hybrid
        self._working_dir = _mkdir(workdir)
        self._source_archive = source_archive

    def filepath_for(self, filename: str, subdir: Optional[str] = None) -> str:
        """
        Returns a path inside the working directory for the provided file.
        """
        if subdir is not None:
            return str(self._working_dir / subdir / filename)
        else:
            return str(self._working_dir / filename)

    def make_subdirectory(self, name: str) -> str:
        """
        Creates a subdirectory in the working dir with name,
        and returns just the name (or "stem") back.
        """
        _mkdir(str(self._working_dir / name))
        return name

    def get_source_archive(self) -> Optional[str]:
        return self._source_archive

    def save_sources(self, relational_data: RelationalData) -> None:
        """
        Creates an archive of all tables' source files and uploads it as a project artifact.
        """
        source_data_dir = relational_data.source_data_handler.dir  # type:ignore
        latest = self._archive_and_upload(
            archive_name=str(self._working_dir / "source_tables.tar.gz"),
            dir_to_archive=source_data_dir,
        )

        # Delete the older version if present (Cloud-only, deletes not supported in Hybrid).
        if (not self._hybrid) and (existing := self._source_archive) is not None:
            self._project.delete_artifact(existing)

        self._source_archive = latest

    def save_backup(self, backup: Backup) -> None:
        """
        Writes backup data as JSON to the working directory,
        uploads the file as a project artifact, and deletes any stale backups.
        """
        backup_path = self._working_dir / "_gretel_backup.json"
        with open(backup_path, "w") as bak:
            json.dump(backup.as_dict, bak)

        # Exit early if hybrid, because this should be a "singleton" project artifact
        # and we cannot delete hybrid project artifacts.
        if self._hybrid:
            return None

        # The backup file does not use the ArtifactCollection because the AC's data
        # (artifact keys) is included in the full backup data, so we would end up
        # "chasing our own tail", for example:
        # - create backup data with AC.backup_key=1, write to file
        # - upload backup file => new backup_key returned from API => AC.backup_key=2
        # Backup data would always be stale and we'd write more backups than we need
        # (we skip uploading backup data if we detect no changes from the latest snapshot).
        latest = self._project.upload_artifact(str(backup_path))
        for artifact in self._project.artifacts:
            key = artifact["key"]
            if key != latest and key.endswith("__gretel_backup.json"):
                self._project.delete_artifact(key)

    def save_debug_summary(self, content: dict) -> None:
        """
        Writes the debug summary content as JSON to the working directory.
        """
        debug_summary_path = self._working_dir / "_gretel_debug_summary.json"
        with open(debug_summary_path, "w") as dbg:
            json.dump(content, dbg)

    def save_classify_outputs(self, filepaths: dict[str, str]) -> None:
        """
        Creates an archive file of the provided classify output files and uploads it
        as a project artifact.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            for item in filepaths.values():
                shutil.copy(item, tmpdir)

            self._archive_and_upload(
                archive_name=str(self._working_dir / "classify_outputs.tar.gz"),
                dir_to_archive=Path(tmpdir),
            )

    def save_transforms_outputs(
        self, filepaths: dict[str, str], run_subdir: str
    ) -> None:
        """
        Archives the run subdirectory and uploads it to the project.
        """
        self._archive_and_upload_run_outputs(run_subdir)

    def save_synthetics_outputs(
        self,
        tables: dict[str, str],
        table_reports: dict[str, dict[str, dict[str, str]]],
        relational_report: Optional[str],
        run_subdir: str,
    ) -> None:
        """
        Archives the run subdirectory and uploads it to the project.
        """
        self._archive_and_upload_run_outputs(run_subdir)

    def _archive_and_upload_run_outputs(self, run_subdir: str) -> None:
        root_dir = self._working_dir / run_subdir
        self._archive_and_upload(
            archive_name=str(root_dir),
            dir_to_archive=root_dir,
        )

    def _archive_and_upload(self, archive_name: str, dir_to_archive: Path) -> str:
        archive_location = shutil.make_archive(
            base_name=archive_name.removesuffix(".tar.gz"),
            format="gztar",
            root_dir=dir_to_archive,
        )
        return self._project.upload_artifact(archive_location)


def _mkdir(name: str) -> Path:
    d = Path(name)
    d.mkdir(parents=True, exist_ok=True)
    return d
