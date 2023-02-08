from __future__ import annotations

import datetime
import json
import logging
import os
import shutil
import tarfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import requests
import smart_open
from gretel_client import configure_session
from gretel_client.projects import Project, create_project, get_project
from gretel_client.projects.jobs import ACTIVE_STATES, END_STATES, Job, Status
from gretel_client.projects.models import Model, read_model_config
from gretel_client.projects.records import RecordHandler

from gretel_trainer.relational.artifacts import ArtifactCollection
from gretel_trainer.relational.backup import (
    Backup,
    BackupForeignKey,
    BackupGenerate,
    BackupGenerateTable,
    BackupRelationalData,
    BackupRelationalDataTable,
    BackupTrain,
    BackupTrainTable,
)
from gretel_trainer.relational.core import (
    MultiTableException,
    RelationalData,
    TableEvaluation,
)
from gretel_trainer.relational.report.report import ReportPresenter, ReportRenderer
from gretel_trainer.relational.sdk_extras import (
    cautiously_refresh_status,
    download_file_artifact,
    download_tar_artifact,
    sqs_score_from_full_report,
)
from gretel_trainer.relational.strategies.ancestral import AncestralStrategy
from gretel_trainer.relational.strategies.independent import IndependentStrategy

GretelModelConfig = Union[str, Path, Dict]

MAX_REFRESH_ATTEMPTS = 3

logger = logging.getLogger(__name__)


class TrainStatus(str, Enum):
    NotStarted = "NotStarted"
    InProgress = "InProgress"
    Completed = "Completed"
    Failed = "Failed"


class GenerateStatus(str, Enum):
    NotStarted = "NotStarted"
    InProgress = "InProgress"
    Completed = "Completed"
    ModelUnavailable = "ModelUnavailable"
    SourcePreserved = "SourcePreserved"
    Failed = "Failed"


@dataclass
class RestoreConfig:
    project: Project
    artifact_collection: ArtifactCollection
    working_dir: Path
    log_message: str
    train_models: Optional[Dict[str, Model]] = None
    training_columns: Optional[Dict[str, List[str]]] = None
    record_size_ratio: Optional[float] = None
    preserved: Optional[List[str]] = None
    record_handlers: Optional[Dict[str, RecordHandler]] = None
    synthetic_output_tables: Optional[Dict[str, pd.DataFrame]] = None


def _create_multitable(
    rel_data: RelationalData, backup: Backup, restore_config: RestoreConfig
) -> MultiTable:
    return MultiTable(
        relational_data=rel_data,
        strategy=backup.strategy,
        gretel_model=backup.gretel_model,
        refresh_interval=backup.refresh_interval,
        restore_config=restore_config,
    )


class MultiTable:
    """
    Relational data support for the Trainer SDK

    Args:
        relational_data (RelationalData): Core data structure representing the source tables and their relationships.
        strategy (str, optional): The strategy to use. Supports "independent" (default) and "ancestral".
        gretel_model (str, optional): The underlying Gretel model to use. Default and acceptable models vary based on strategy.
        project_display_name (str, optional): Display name in the console for a new Gretel project holding models and artifacts. Defaults to "multi-table".
        refresh_interval (int, optional): Frequency in seconds to poll Gretel Cloud for job statuses. Must be at least 30. Defaults to 60 (1m).
        restore_config (RestoreConfig, optional): Data used to restore from backup. Should not be supplied manually; instead use the `restore` classmethod.
    """

    def __init__(
        self,
        relational_data: RelationalData,
        *,
        strategy: str = "independent",
        gretel_model: Optional[str] = None,
        project_display_name: Optional[str] = None,
        refresh_interval: Optional[int] = None,
        restore_config: Optional[RestoreConfig] = None,
        # deprecated
        project_name: Optional[str] = None,
        working_dir: Optional[str] = None,
    ):
        # Deprecation warnings
        if project_name is not None:
            logger.warning(
                "The 'project_name' parameter is deprecated and will be removed in a future release. Please use 'project_display_name' instead."
            )
            project_display_name = project_display_name or project_name
        if working_dir is not None:
            logger.warning(
                "The 'working_dir' parameter is deprecated and will be removed in a future release. Local working directory names will always match the project's unique name."
            )

        self._strategy = _validate_strategy(strategy)
        model_name, model_config = self._validate_gretel_model(gretel_model)
        self._gretel_model = model_name
        self._model_config = model_config
        self._set_refresh_interval(refresh_interval)

        self.relational_data = relational_data
        self._artifact_collection = ArtifactCollection()
        self._latest_backup: Optional[Backup] = None
        self._models: Dict[str, Model] = {}
        self._record_handlers: Dict[str, RecordHandler] = {}
        self._record_size_ratio = 1.0
        self.train_statuses: Dict[str, TrainStatus] = {}
        self._training_columns: Dict[str, List[str]] = {}
        self._reset_train_statuses(self.relational_data.list_all_tables())
        self._reset_generation_statuses()
        self.synthetic_output_tables: Dict[str, pd.DataFrame] = {}
        self.evaluations = defaultdict(lambda: TableEvaluation())

        if restore_config is None:
            configure_session(api_key="prompt", cache="yes", validate=True)
            project_display_name = project_display_name or "multi-table"
            self._project = create_project(display_name=project_display_name)
            logger.info(
                f"Created project `{self._project.display_name}` with unique name `{self._project.name}`."
            )
            self._working_dir = _mkdir(self._project.name)
            self._create_debug_summary()
            self._upload_sources_to_project()
        else:
            self._project = restore_config.project
            logger.info(
                f"Connected to existing project `{self._project.display_name}` with unique name `{self._project.name}`."
            )
            self._working_dir = restore_config.working_dir
            self._artifact_collection = restore_config.artifact_collection
            self._models = restore_config.train_models or self._models
            for table, model in self._models.items():
                train_status = _train_status_for_model(model)
                self.train_statuses[table] = train_status
                if train_status == TrainStatus.Completed:
                    self._strategy.update_evaluation_from_model(
                        table, self.evaluations, model, self._working_dir
                    )
            self._record_handlers = (
                restore_config.record_handlers or self._record_handlers
            )
            self._record_size_ratio = (
                restore_config.record_size_ratio or self._record_size_ratio
            )
            self._training_columns = (
                restore_config.training_columns or self._training_columns
            )
            if restore_config.preserved is not None:
                for table in restore_config.preserved:
                    self.generate_statuses[table] = GenerateStatus.SourcePreserved
            if restore_config.record_handlers is not None:
                for table, record_handler in restore_config.record_handlers.items():
                    self.generate_statuses[table] = _generate_status_for_record_handler(
                        record_handler
                    )
            synth_out_tables = restore_config.synthetic_output_tables
            if synth_out_tables is not None:
                self.synthetic_output_tables = synth_out_tables
                for table in synth_out_tables:
                    self._attach_existing_reports(table)

            logger.info(restore_config.log_message)

    @classmethod
    def restore(cls, backup_file: str) -> MultiTable:
        logger.info(f"Restoring from backup file `{backup_file}`.")
        with open(backup_file, "r") as b:
            backup = Backup.from_dict(json.load(b))

        configure_session(api_key="prompt", cache="yes", validate=True)
        project_name = backup.project_name
        # Raises GretelProjectEror if not found
        project = get_project(name=project_name)
        working_dir = _mkdir(project_name)
        restore_config = RestoreConfig(
            project=project,
            artifact_collection=backup.artifact_collection,
            working_dir=working_dir,
            log_message="",
        )

        artifact_collection = backup.artifact_collection

        debug_summary_id = artifact_collection.gretel_debug_summary
        if debug_summary_id is not None:
            debug_summary_path = working_dir / "_gretel_debug_summary.json"
            download_file_artifact(project, debug_summary_id, debug_summary_path)

        source_archive_id = artifact_collection.source_archive
        if source_archive_id is None:
            raise MultiTableException(
                "Cannot restore from backup: source archive is missing."
            )
        source_archive_path = working_dir / "source_tables.tar.gz"
        download_tar_artifact(project, source_archive_id, source_archive_path)
        with tarfile.open(source_archive_path, "r:gz") as tar:
            tar.extractall()

        # Restore RelationalData instance
        rel_data = RelationalData()
        for table_name, table_backup in backup.relational_data.tables.items():
            local_source_path = working_dir / f"source_{table_name}.csv"
            source_data = pd.read_csv(str(local_source_path))
            rel_data.add_table(
                name=table_name, primary_key=table_backup.primary_key, data=source_data
            )
        for fk_backup in backup.relational_data.foreign_keys:
            rel_data.add_foreign_key(
                foreign_key=fk_backup.foreign_key, referencing=fk_backup.referencing
            )

        backup_train = backup.train
        if backup_train is None:
            restore_config.log_message = "No model train data present in backup. From here, your next step is to call `train`."
            return _create_multitable(rel_data, backup, restore_config)

        train_models: Dict[str, Model] = {}
        training_columns: Dict[str, List[str]] = {}
        failed_to_train = []
        for table_name, table_train_backup in backup_train.tables.items():
            model = project.get_model(table_train_backup.model_id)
            train_models[table_name] = model
            training_columns[table_name] = table_train_backup.training_columns

            train_status = _train_status_for_model(model)
            if train_status == TrainStatus.Failed:
                failed_to_train.append(table_name)
            elif train_status == TrainStatus.InProgress:
                logger.warning(
                    f"Training still in progress for table `{table_name}`. From here, your next step is to wait for training to finish, and re-attempt restoring from backup once all models have completed training. You can view training progress in the Console under the `{table_name}` model page in the `{project.display_name} ({project.name})` project."
                )
                raise MultiTableException(
                    "Cannot restore while model training is actively in progress."
                )
            else:
                download_file_artifact(
                    project,
                    model.data_source,
                    working_dir / f"synthetics_train_{table_name}.csv",
                )
                logger.info(
                    f"Restored model for `{table_name}` with status {train_status}."
                )
        restore_config.train_models = train_models
        restore_config.training_columns = training_columns
        if len(failed_to_train) > 0:
            restore_config.log_message = f"Training failed for tables: {failed_to_train}. From here, your next step is to try retraining them with modified data by calling `retrain_tables`."
            return _create_multitable(rel_data, backup, restore_config)
        else:
            logger.info("Training restoration complete.")

        backup_generate = backup.generate
        if backup_generate is None:
            restore_config.log_message = "No generation data present in backup. From here, your next step is to call `generate`."
            return _create_multitable(rel_data, backup, restore_config)

        backup_copy_dest = f"{backup_file}.restoring"
        logger.info(
            f"Found backup generate data. Copying this backup file to `{backup_copy_dest}` since restoring generate data can alter the backup file."
        )
        shutil.copy(backup_file, backup_copy_dest)

        restore_config.record_size_ratio = backup_generate.record_size_ratio
        restore_config.preserved = backup_generate.preserved

        record_handlers: Dict[str, RecordHandler] = {}
        for table_name, table_generate_backup in backup_generate.tables.items():
            record_handler = train_models[table_name].get_record_handler(
                table_generate_backup.record_handler_id
            )
            record_handlers[table_name] = record_handler
            data_source = record_handler.data_source
            if data_source is not None:
                out_path = working_dir / f"synthetics_seed_{table_name}.csv"
                download_file_artifact(project, data_source, out_path)

        restore_config.record_handlers = record_handlers

        synthetics_output_archive_id = artifact_collection.synthetics_outputs_archive
        if synthetics_output_archive_id is None:
            restore_config.log_message = f"At time of last backup, generation had not yet finished. From here, you can attempt to resume that job via `generate(resume=True)`, or restart generation from scratch via a regular call to `generate`."
        else:
            synthetics_output_archive_path = (
                working_dir / "synthetics_output_tables.tar.gz"
            )
            download_tar_artifact(
                project, synthetics_output_archive_id, synthetics_output_archive_path
            )
            with tarfile.open(synthetics_output_archive_path, "r:gz") as tar:
                tar.extractall()
            synthetic_output_tables: Dict[str, pd.DataFrame] = {}
            for table_name in backup.relational_data.tables:
                local_synth_path = working_dir / f"synth_{table_name}.csv"
                synthetic_output_tables[table_name] = pd.read_csv(str(local_synth_path))
            restore_config.synthetic_output_tables = synthetic_output_tables
            restore_config.log_message = "Generation jobs for all tables from previous run finished prior to backup. From here, you can access your synthetic data as Pandas DataFrames via `synthetic_output_tables`, or review them in CSV format along with the relational report in the local working directory."

        return _create_multitable(rel_data, backup, restore_config)

    def _backup(self) -> None:
        backup = self._build_backup()
        # exit early if nothing has changed since last backup
        if backup == self._latest_backup:
            return None

        # write to local directory
        backup_path = self._working_dir / "_gretel_backup.json"
        with open(backup_path, "w") as bak:
            json.dump(backup.as_dict, bak)

        _upload_gretel_backup(self._project, str(backup_path))

        self._latest_backup = backup

    def _build_backup(self) -> Backup:
        # Relational Data
        tables = {}
        foreign_keys = []
        for table in self.relational_data.list_all_tables():
            tables[table] = BackupRelationalDataTable(
                primary_key=self.relational_data.get_primary_key(table),
            )
            foreign_keys.extend(
                [
                    BackupForeignKey.from_fk(key)
                    for key in self.relational_data.get_foreign_keys(table)
                ]
            )
        backup_relational_data = BackupRelationalData(
            tables=tables, foreign_keys=foreign_keys
        )

        backup = Backup(
            project_name=self._project.name,
            strategy=self._strategy.name,
            gretel_model=self._gretel_model,
            working_dir=str(self._working_dir),
            refresh_interval=self._refresh_interval,
            artifact_collection=replace(self._artifact_collection),
            relational_data=backup_relational_data,
        )

        # Train
        if len(self._models) > 0:
            backup_train_tables = {
                table: BackupTrainTable(
                    model_id=model.model_id,
                    training_columns=self._training_columns.get(table, []),
                )
                for table, model in self._models.items()
            }
            backup.train = BackupTrain(tables=backup_train_tables)

        # Generate
        if len(self._record_handlers) > 0:
            backup_generate_tables = {
                table: BackupGenerateTable(
                    record_handler_id=rh.record_id,
                )
                for table, rh in self._record_handlers.items()
            }
            preserved = [
                table
                for table, status in self.generate_statuses.items()
                if status == GenerateStatus.SourcePreserved
            ]
            backup.generate = BackupGenerate(
                tables=backup_generate_tables,
                preserved=preserved,
                record_size_ratio=self._record_size_ratio,
            )

        return backup

    def _set_refresh_interval(self, interval: Optional[int]) -> None:
        if interval is None:
            self._refresh_interval = 60
        else:
            if interval < 30:
                logger.warning(
                    "Refresh interval must be at least 30 seconds. Setting to 30."
                )
                self._refresh_interval = 30
            else:
                self._refresh_interval = interval

    def _create_debug_summary(self) -> None:
        debug_summary_path = self._working_dir / "_gretel_debug_summary.json"
        content = {
            "relational_data": self.relational_data.debug_summary(),
            "strategy": self._strategy.name,
            "model": self._gretel_model,
        }
        with open(debug_summary_path, "w") as dbg:
            json.dump(content, dbg)
        self._artifact_collection.upload_gretel_debug_summary(
            self._project, str(debug_summary_path)
        )

    def transform(
        self,
        configs: Dict[str, GretelModelConfig],
        in_place: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Applies supplied transform model configurations to tables. Returned dictionary includes all transformed
        tables, which may not include all known tables (i.e. if a transform config was not provided).

        Args:
            configs (dict[str, GretelModelConfig]): keys are table names and values are Transform model configs.
            in_place (bool, optional): If True, overwrites internal source dataframes with transformed dataframes,
            which means subsequent synthetic model training would be performed on the transform results.

        Returns:
            dict[str, pd.DataFrame]: keys are table names and values are transformed tables
        """
        output_tables = self._execute_transform_jobs(configs)
        output_tables = self._strategy.label_encode_keys(
            self.relational_data, output_tables
        )

        if in_place:
            for table_name, transformed_table in output_tables.items():
                self.relational_data.update_table_data(table_name, transformed_table)

        self.transform_output_tables = output_tables
        return self.transform_output_tables

    def _execute_transform_jobs(
        self, configs: Dict[str, GretelModelConfig]
    ) -> Dict[str, pd.DataFrame]:
        # Ensure consistent, friendly names in the console
        named_configs = {}
        for table_name, config in configs.items():
            config_dict = read_model_config(config)
            config_dict["name"] = f"{table_name}-transforms"
            named_configs[table_name] = config_dict

        output_tables: Dict[str, pd.DataFrame] = {}
        models: Dict[str, Model] = {}
        record_handlers: Dict[str, RecordHandler] = {}
        model_statuses: Dict[str, Status] = {}
        record_handler_statuses: Dict[str, Status] = {}
        refresh_attempts: Dict[str, int] = defaultdict(int)

        # Create all models for training
        for table_name, config in named_configs.items():
            self._log_start(table_name, "transforms model training")
            table_data = self.relational_data.get_table_data(table_name)
            model = self._project.create_model_obj(
                model_config=config, data_source=table_data
            )
            model.submit_cloud()
            models[table_name] = model

        def _more_to_do() -> bool:
            return len(record_handler_statuses) != len(named_configs) or not all(
                status in END_STATES for status in record_handler_statuses.values()
            )

        while _more_to_do():
            self._wait_refresh_interval()

            for table_name in named_configs:
                # No need to re-check tables that are totally finished
                if record_handler_statuses.get(table_name) in END_STATES:
                    continue

                # If we consistently failed to refresh the job status, fail the table
                if refresh_attempts[table_name] >= MAX_REFRESH_ATTEMPTS:
                    self._log_lost_contact(table_name)
                    record_handler_statuses[table_name] = Status.LOST
                    continue

                # If RH is not finished but model training is, update RH status and handle
                if model_statuses.get(table_name) in END_STATES:
                    record_handler = record_handlers[table_name]
                    rh_status = cautiously_refresh_status(
                        record_handler, table_name, refresh_attempts
                    )
                    record_handler_statuses[table_name] = rh_status

                    if rh_status == Status.COMPLETED:
                        self._log_success(table_name, "transforms data generation")
                        out_table = _get_data_from_record_handler(record_handler)
                        output_tables[table_name] = out_table
                    elif rh_status in END_STATES:
                        self._log_failed(table_name, "transforms data generation")
                    else:
                        self._log_in_progress(
                            table_name, rh_status, "transforms data generation"
                        )

                    continue

                # Here = model training was last seen in progress. Update model status and handle.
                model = models[table_name]
                model_status = cautiously_refresh_status(
                    model, table_name, refresh_attempts
                )
                model_statuses[table_name] = model_status

                if model_status == Status.COMPLETED:
                    self._log_success(table_name, "transforms model training")
                    self._log_start(table_name, "transforms data generation")
                    table_data = self.relational_data.get_table_data(table_name)
                    rh = model.create_record_handler_obj(data_source=table_data)
                    rh.submit_cloud()
                    record_handlers[table_name] = rh
                elif model_status in END_STATES:
                    self._log_failed(table_name, "transforms model training")
                    # Set a terminal RH status for this table so we don't keep checking it.
                    # In this case, CANCELLED is standing in for "Won't Attempt"
                    record_handler_statuses[table_name] = Status.CANCELLED
                else:
                    self._log_in_progress(
                        table_name, model_status, "transforms model training"
                    )

        return output_tables

    def _prepare_training_data(self, tables: List[str]) -> Dict[str, Path]:
        """
        Exports a copy of each table prepared for training by the configured strategy
        to the working directory. Returns a dict with table names as keys and Paths
        to the CSVs as values.
        """
        training_data = self._strategy.prepare_training_data(self.relational_data)
        for table, df in training_data.items():
            self._training_columns[table] = list(df.columns)
        training_paths = {}

        for table_name in tables:
            training_path = self._working_dir / f"synthetics_train_{table_name}.csv"
            training_data[table_name].to_csv(training_path, index=False)
            training_paths[table_name] = training_path

        return training_paths

    def _table_model_config(self, table_name: str) -> Dict:
        config_dict = read_model_config(self._model_config)
        config_dict["name"] = table_name
        return config_dict

    def _train_models(self, training_data: Dict[str, Path]) -> None:
        for table_name, training_csv in training_data.items():
            self._log_start(table_name, "model training")
            self.train_statuses[table_name] = TrainStatus.InProgress
            table_model_config = self._table_model_config(table_name)
            model = self._project.create_model_obj(
                model_config=table_model_config, data_source=str(training_csv)
            )
            model.submit_cloud()
            self._models[table_name] = model

        self._backup()

        refresh_attempts: Dict[str, int] = defaultdict(int)

        def _more_to_do() -> bool:
            return any(
                [
                    status == TrainStatus.InProgress
                    for status in self.train_statuses.values()
                ]
            )

        while _more_to_do():
            self._wait_refresh_interval()

            for table_name in training_data:
                # No need to do anything with tables in terminal state
                if self.train_statuses[table_name] in (
                    TrainStatus.Completed,
                    TrainStatus.Failed,
                ):
                    continue

                # If we consistently failed to refresh the job status, fail the table
                if refresh_attempts[table_name] >= MAX_REFRESH_ATTEMPTS:
                    self._log_lost_contact(table_name)
                    self.train_statuses[table_name] = TrainStatus.Failed
                    continue

                model = self._models[table_name]

                status = cautiously_refresh_status(model, table_name, refresh_attempts)

                if status == Status.COMPLETED:
                    self._log_success(table_name, "model training")
                    self.train_statuses[table_name] = TrainStatus.Completed
                    self._strategy.update_evaluation_from_model(
                        table_name, self.evaluations, model, self._working_dir
                    )
                elif status in END_STATES:
                    # already checked explicitly for completed; all other end states are effectively failures
                    self._log_failed(table_name, "model training")
                    self.train_statuses[table_name] = TrainStatus.Failed
                else:
                    self._log_in_progress(table_name, status, "model training")
                    continue

            self._backup()

    def train(self) -> None:
        """Train synthetic data models on each table in the relational dataset"""
        tables = self.relational_data.list_all_tables()
        self._reset_train_statuses(tables)

        training_data = self._prepare_training_data(tables)
        self._train_models(training_data)

    def retrain_tables(self, tables: Dict[str, pd.DataFrame]) -> None:
        """
        Provide updated table data and retrain. This method overwrites the table data in the
        `RelationalData` instance. It should be used when initial training fails and source data
        needs to be altered, but progress on other tables can be left as-is.
        """
        for table_name, table_data in tables.items():
            self.relational_data.update_table_data(table_name, table_data)

        self._upload_sources_to_project()

        tables_to_retrain = self._strategy.tables_to_retrain(
            list(tables.keys()), self.relational_data
        )

        self._reset_train_statuses(tables_to_retrain)
        training_data = self._prepare_training_data(tables_to_retrain)
        self._train_models(training_data)

    def _upload_sources_to_project(self) -> None:
        archive_path = self._working_dir / "source_tables.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            for table in self.relational_data.list_all_tables():
                out_path = self._working_dir / f"source_{table}.csv"
                df = self.relational_data.get_table_data(table)
                df.to_csv(out_path, index=False)
                tar.add(out_path)
        self._artifact_collection.upload_source_archive(
            self._project, str(archive_path)
        )
        self._backup()

    def _reset_train_statuses(self, tables: List[str]) -> None:
        for table in tables:
            self.train_statuses[table] = TrainStatus.NotStarted

    def generate(
        self,
        record_size_ratio: Optional[float] = None,
        preserve_tables: Optional[List[str]] = None,
        resume: bool = False,
    ) -> None:
        """
        Sample synthetic data from trained models.
        Tables that did not train successfully will be omitted from the output dictionary.
        Tables listed in `preserve_tables` *may differ* from source tables in foreign key columns, to ensure
        joining to parent tables (which may have been synthesized) continues to work properly.

        Args:
            record_size_ratio (float, optional): Ratio to upsample real world data size with. Defaults to 1.
            preserve_tables (list[str], optional): List of tables to skip sampling and leave (mostly) identical to source.
            resume (bool, optional): Set to True when restoring from a backup to complete a previous, interrupted run.

        Returns:
            dict[str, pd.DataFrame]: Return a dictionary of table names and output data.
        """
        output_tables = {}

        if resume:
            if record_size_ratio is not None:
                logger.warning(
                    "Cannot set record_size_ratio when resuming previous generation. Ignoring."
                )
            if preserve_tables is not None:
                logger.warning(
                    "Cannot set preserve_tables when resuming previous generation. Ignoring."
                )
            preserve_tables = [
                table
                for table, status in self.generate_statuses.items()
                if status == GenerateStatus.SourcePreserved
            ]
            for table_name, record_handler in self._record_handlers.items():
                # Reset statuses of completed record handlers to usher them (immediately)
                # through post-processing. Note that in ancestral strategy, this assumes
                # seed generation is deterministic, because child tables may be in progress
                # and we'd need the seed they were started with to be equivalent to the seed
                # we "would" generate from the restored, post-processed parent.
                if self.generate_statuses[table_name] == GenerateStatus.Completed:
                    self.generate_statuses[table_name] = GenerateStatus.NotStarted
        else:
            if record_size_ratio is not None:
                self._record_size_ratio = record_size_ratio
            self._reset_generation_statuses()
            preserve_tables = preserve_tables or []
            self._strategy.validate_preserved_tables(
                preserve_tables, self.relational_data
            )
            self._record_handlers: Dict[str, RecordHandler] = {}

        self._skip_some_tables(preserve_tables, output_tables)
        all_tables = self.relational_data.list_all_tables()
        refresh_attempts: Dict[str, int] = defaultdict(int)

        def _more_to_do() -> bool:
            return not all(
                [
                    self._table_generation_in_terminal_state(table)
                    for table in all_tables
                ]
            )

        first_pass = True
        while _more_to_do():
            # Don't wait needlessly the first time through.
            if first_pass:
                first_pass = False
            else:
                self._wait_refresh_interval()

            for table_name, record_handler in self._record_handlers.items():
                # No need to do anything with tables in terminal state
                if self._table_generation_in_terminal_state(table_name):
                    continue

                # If we consistently failed to refresh the job via API, fail the table
                if refresh_attempts[table_name] >= MAX_REFRESH_ATTEMPTS:
                    self._log_lost_contact(table_name)
                    self.generate_statuses[table_name] = GenerateStatus.Failed
                    continue

                status = cautiously_refresh_status(
                    record_handler, table_name, refresh_attempts
                )

                if status == Status.COMPLETED:
                    self._log_success(table_name, "synthetic data generation")
                    self.generate_statuses[table_name] = GenerateStatus.Completed
                    record_handler_result = _get_data_from_record_handler(
                        record_handler
                    )
                    output_tables[
                        table_name
                    ] = self._strategy.post_process_individual_synthetic_result(
                        table_name, self.relational_data, record_handler_result
                    )
                elif status in END_STATES:
                    # already checked explicitly for completed; all other end states are effectively failures
                    self._log_failed(table_name, "synthetic data generation")
                    self.generate_statuses[table_name] = GenerateStatus.Failed
                else:
                    self._log_in_progress(
                        table_name, status, "synthetic data generation"
                    )
                    continue

            # Determine if we can start any more jobs
            in_progress_tables = [
                table
                for table in all_tables
                if self._table_generation_in_progress(table)
            ]
            finished_tables = [
                table
                for table in all_tables
                if self._table_generation_in_terminal_state(table)
            ]

            ready_tables = self._strategy.ready_to_generate(
                self.relational_data, in_progress_tables, finished_tables
            )

            for table_name in ready_tables:
                table_job = self._strategy.get_generation_job(
                    table_name,
                    self.relational_data,
                    self._record_size_ratio,
                    output_tables,
                    self._working_dir,
                    self._training_columns[table_name],
                )
                self._log_start(table_name, "synthetic data generation")
                self.generate_statuses[table_name] = GenerateStatus.InProgress
                model = self._models[table_name]
                record_handler = model.create_record_handler_obj(**table_job)
                record_handler.submit_cloud()
                self._record_handlers[table_name] = record_handler

            self._backup()

        output_tables = self._strategy.post_process_synthetic_results(
            output_tables, preserve_tables, self.relational_data
        )

        tables_with_incomplete_evaluations = {
            table: df
            for table, df in output_tables.items()
            if not self.evaluations[table].is_complete()
        }
        self._expand_evaluations(tables_with_incomplete_evaluations)

        logger.info("Creating relational report")
        self.create_relational_report()

        archive_path = self._working_dir / "synthetics_outputs.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(self._working_dir / "relational_report.html")
            for table in self.relational_data.list_all_tables():
                # Add synthetic output table
                synthetic_df = output_tables[table]
                out_path = self._working_dir / f"synth_{table}.csv"
                synthetic_df.to_csv(out_path, index=False)
                tar.add(out_path)
                # Add individual and cross_table reports
                for eval_type in ["individual", "cross_table"]:
                    for ext in ["html", "json"]:
                        filename = f"synthetics_{eval_type}_evaluation_{table}.{ext}"
                        try:
                            tar.add(self._working_dir / filename)
                        except FileNotFoundError:
                            logger.warning(
                                f"Could not find `{filename}` in working directory"
                            )

        self._artifact_collection.upload_synthetics_outputs_archive(
            self._project, str(archive_path)
        )
        self._backup()
        self.synthetic_output_tables = output_tables

    def _expand_evaluations(self, output_tables: Dict[str, pd.DataFrame]) -> None:
        """
        Adds evaluation metrics for the "opposite" correlation strategy using the Gretel Evaluate API.
        """
        for table_name in output_tables:
            self._strategy.update_evaluation_via_evaluate(
                evaluation=self.evaluations[table_name],
                table=table_name,
                rel_data=self.relational_data,
                synthetic_tables=output_tables,
                working_dir=self._working_dir,
            )

    def create_relational_report(self) -> None:
        presenter = ReportPresenter(
            rel_data=self.relational_data,
            evaluations=self.evaluations,
            now=datetime.datetime.utcnow(),
        )
        output_path = self._working_dir / "relational_report.html"
        with open(output_path, "w") as report:
            html_content = ReportRenderer().render(presenter)
            report.write(html_content)

    def _reset_generation_statuses(self) -> None:
        """
        Sets the GenerateStatus for all known tables to NotStarted.
        """
        self.generate_statuses = {
            table_name: GenerateStatus.NotStarted
            for table_name in self.relational_data.list_all_tables()
        }

    def _skip_some_tables(
        self, preserve_tables: List[str], output_tables: Dict[str, pd.DataFrame]
    ) -> None:
        "Updates state for tables being preserved and tables lacking trained models."
        for table in self.relational_data.list_all_tables():
            if table in preserve_tables:
                self.generate_statuses[table] = GenerateStatus.SourcePreserved
                output_tables[table] = self._strategy.get_preserved_data(
                    table, self.relational_data
                )
            elif self.train_statuses[table] != TrainStatus.Completed:
                logger.info(
                    f"Skipping synthetic data generation for `{table}` because it does not have a trained model"
                )
                self.generate_statuses[table] = GenerateStatus.ModelUnavailable
                for descendant in self.relational_data.get_descendants(table):
                    logger.info(
                        f"Skipping synthetic data generation for `{descendant}` because it depends on `{table}`"
                    )
                    self.generate_statuses[descendant] = GenerateStatus.ModelUnavailable

    def _table_generation_in_progress(self, table: str) -> bool:
        return self.generate_statuses[table] == GenerateStatus.InProgress

    def _table_generation_in_terminal_state(self, table: str) -> bool:
        return self.generate_statuses[table] in [
            GenerateStatus.Completed,
            GenerateStatus.SourcePreserved,
            GenerateStatus.ModelUnavailable,
            GenerateStatus.Failed,
        ]

    def _attach_existing_reports(self, table: str) -> None:
        individual_path = (
            self._working_dir / f"synthetics_individual_evaluation_{table}.json"
        )
        cross_table_path = (
            self._working_dir / f"synthetics_cross_table_evaluation_{table}.json"
        )

        individual_report_json = json.loads(smart_open.open(individual_path).read())
        cross_table_report_json = json.loads(smart_open.open(individual_path).read())

        self.evaluations[table].individual_report_json = individual_report_json
        self.evaluations[table].individual_sqs = sqs_score_from_full_report(
            individual_report_json
        )
        self.evaluations[table].cross_table_report_json = cross_table_report_json
        self.evaluations[table].cross_table_sqs = sqs_score_from_full_report(
            cross_table_report_json
        )

    def _wait_refresh_interval(self) -> None:
        logger.info(f"Next status check in {self._refresh_interval} seconds.")
        time.sleep(self._refresh_interval)

    def _log_start(self, table_name: str, action: str) -> None:
        logger.info(f"Starting {action} for `{table_name}`.")

    def _log_in_progress(self, table_name: str, status: Status, action: str) -> None:
        logger.info(
            f"{action.capitalize()} job for `{table_name}` still in progress (status: {status})."
        )

    def _log_failed(self, table_name: str, action: str) -> None:
        logger.info(f"{action.capitalize()} failed for `{table_name}`.")

    def _log_success(self, table_name: str, action: str) -> None:
        logger.info(f"{action.capitalize()} successfully completed for `{table_name}`.")

    def _log_lost_contact(self, table_name: str) -> None:
        logger.warning(f"Lost contact with job for `{table_name}`.")

    def _validate_gretel_model(self, gretel_model: Optional[str]) -> Tuple[str, str]:
        gretel_model = (gretel_model or self._strategy.default_model).lower()
        supported_models = self._strategy.supported_models
        if gretel_model not in supported_models:
            msg = f"Invalid gretel model requested: {gretel_model}. The selected strategy supports: {supported_models}."
            logger.warning(msg)
            raise MultiTableException(msg)

        _BLUEPRINTS = {
            "amplify": "synthetics/amplify",
            "actgan": "synthetics/tabular-actgan",
            "lstm": "synthetics/tabular-lstm",
        }

        return (gretel_model, _BLUEPRINTS[gretel_model])


def _get_data_from_record_handler(record_handler: RecordHandler) -> pd.DataFrame:
    return pd.read_csv(record_handler.get_artifact_link("data"), compression="gzip")


def _validate_strategy(strategy: str) -> Union[IndependentStrategy, AncestralStrategy]:
    strategy = strategy.lower()

    if strategy == "single-table":
        logger.warning(
            "The 'single-table' value for the 'strategy' parameter is deprecated and will be removed in a future release. Please use 'independent' instead."
        )
        return IndependentStrategy()
    elif strategy == "independent":
        return IndependentStrategy()
    elif strategy == "cross-table":
        logger.warning(
            "The 'cross-table' value for the 'strategy' parameter is deprecated and will be removed in a future release. Please use 'ancestral' instead."
        )
        return AncestralStrategy()
    elif strategy == "ancestral":
        return AncestralStrategy()
    else:
        msg = f"Unrecognized strategy requested: {strategy}. Supported strategies are `independent` and `ancestral`."
        logger.warning(msg)
        raise MultiTableException(msg)


def _train_status_for_model(model: Model) -> TrainStatus:
    if model.status == Status.COMPLETED:
        return TrainStatus.Completed
    elif model.status in END_STATES:
        return TrainStatus.Failed
    elif model.status in ACTIVE_STATES:
        return TrainStatus.InProgress
    else:
        return TrainStatus.NotStarted


def _generate_status_for_record_handler(rh: RecordHandler) -> GenerateStatus:
    if rh.status == Status.COMPLETED:
        return GenerateStatus.Completed
    elif rh.status in END_STATES:
        return GenerateStatus.Failed
    elif rh.status in ACTIVE_STATES:
        return GenerateStatus.InProgress
    else:
        return GenerateStatus.NotStarted


def _mkdir(name: str) -> Path:
    d = Path(name)
    os.makedirs(d, exist_ok=True)
    return d


def _upload_gretel_backup(project: Project, path: str) -> None:
    latest = project.upload_artifact(path)
    for artifact in project.artifacts:
        key = artifact["key"]
        if key != latest and key.endswith("__gretel_backup.json"):
            project.delete_artifact(key)
