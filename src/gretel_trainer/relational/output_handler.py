import json
import logging
import tarfile

from pathlib import Path
from typing import Optional, Protocol

from gretel_client.projects import Project
from gretel_trainer.relational.artifacts import (
    archive_items,
    archive_nested_dir,
    ArtifactCollection,
)
from gretel_trainer.relational.backup import Backup
from gretel_trainer.relational.core import RelationalData, Scope

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

    def save_synthetics_training_files(self, filepaths: dict[str, str]) -> None:
        """
        Callback when synthetic preprocessing completes to persist training data (model `data_source`s).
        """
        ...

    def save_synthetics_outputs(
        self,
        tables: dict[str, str],
        table_reports: dict[str, dict[str, dict[str, str]]],
        relational_report: str,
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
        artifact_collection: ArtifactCollection,
    ):
        self._project = project
        self._hybrid = hybrid
        self._artifact_collection = artifact_collection
        self._working_dir = _mkdir(workdir)

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

    def save_sources(self, relational_data: RelationalData) -> None:
        """
        Creates an archive of all tables' source files and uploads it as a project artifact.
        """
        logger.info("Uploading initial configuration state to project.")
        archive_path = self._working_dir / "source_tables.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            for table in relational_data.list_all_tables(Scope.ALL):
                source_path = Path(relational_data.get_table_source(table))
                filename = source_path.name
                tar.add(source_path, arcname=f"source_{filename}")
        self._artifact_collection.upload_source_archive(
            self._project, str(archive_path)
        )

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
        Writes the debug summary content as JSON to the working directory and
        uploads it as a project artifact.
        """
        debug_summary_path = self._working_dir / "_gretel_debug_summary.json"
        with open(debug_summary_path, "w") as dbg:
            json.dump(content, dbg)
        self._artifact_collection.upload_gretel_debug_summary(
            self._project, str(debug_summary_path)
        )

    def save_classify_outputs(self, filepaths: dict[str, str]) -> None:
        """
        Creates an archive file of the provided classify output files and uploads it
        as a project artifact.
        """
        archive_path = self._working_dir / "classify_outputs.tar.gz"
        archive_items(archive_path, list(filepaths.values()))
        self._artifact_collection.upload_classify_outputs_archive(
            self._project, str(archive_path)
        )

    def save_transforms_outputs(
        self, filepaths: dict[str, str], run_subdir: str
    ) -> None:
        """
        Adds the entire run subdirectory to the aggregate transforms_outputs archive file
        (or creates that archive if it does not yet exist) and uploads the archive file
        as a project artifact.
        """
        all_runs_archive_path = self._working_dir / "transforms_outputs.tar.gz"

        archive_nested_dir(
            targz=all_runs_archive_path,
            directory=self._working_dir / run_subdir,
            name=run_subdir,
        )

        self._artifact_collection.upload_transforms_outputs_archive(
            self._project, str(all_runs_archive_path)
        )

    def save_synthetics_training_files(self, filepaths: dict[str, str]) -> None:
        """
        Creates an archive of the provided pre-processed synthetics training data source files
        and uploads it as a project artifact.
        """
        archive_path = self._working_dir / "synthetics_training.tar.gz"
        archive_items(archive_path, list(filepaths.values()))
        self._artifact_collection.upload_synthetics_training_archive(
            self._project, str(archive_path)
        )

    def save_synthetics_outputs(
        self,
        tables: dict[str, str],
        table_reports: dict[str, dict[str, dict[str, str]]],
        relational_report: str,
        run_subdir: str,
    ) -> None:
        """
        Adds the entire run subdirectory to the aggregate synthetics_outputs archive file
        (or creates that archive if it does not yet exist) and uploads the archive file
        as a project artifact.
        """
        all_runs_archive_path = self._working_dir / "synthetics_outputs.tar.gz"

        archive_nested_dir(
            targz=all_runs_archive_path,
            directory=self._working_dir / run_subdir,
            name=run_subdir,
        )

        self._artifact_collection.upload_synthetics_outputs_archive(
            self._project, str(all_runs_archive_path)
        )


def _mkdir(name: str) -> Path:
    d = Path(name)
    d.mkdir(parents=True, exist_ok=True)
    return d
