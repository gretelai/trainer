from __future__ import annotations

import json
import logging
import os
import shutil
import tarfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import datetime
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

from gretel_trainer.relational.artifacts import ArtifactCollection, add_to_tar
from gretel_trainer.relational.backup import (
    Backup,
    BackupForeignKey,
    BackupGenerate,
    BackupRelationalData,
    BackupSyntheticsTrain,
    BackupTransformsTrain,
)
from gretel_trainer.relational.core import (
    GretelModelConfig,
    MultiTableException,
    RelationalData,
    TableEvaluation,
)
from gretel_trainer.relational.model_config import (
    make_evaluate_config,
    make_synthetics_config,
    make_transform_config,
)
from gretel_trainer.relational.report.report import ReportPresenter, ReportRenderer
from gretel_trainer.relational.sdk_extras import (
    cautiously_refresh_status,
    delete_data_source,
    download_file_artifact,
    download_tar_artifact,
    get_job_id,
    room_in_project,
    sqs_score_from_full_report,
)
from gretel_trainer.relational.strategies.ancestral import AncestralStrategy
from gretel_trainer.relational.strategies.independent import IndependentStrategy
from gretel_trainer.relational.task_runner import run_task
from gretel_trainer.relational.tasks import (
    SyntheticsEvaluateTask,
    SyntheticsRunTask,
    SyntheticsTrainTask,
    TransformsRunTask,
    TransformsTrainTask,
)
from gretel_trainer.relational.workflow_state import (
    SyntheticsRun,
    SyntheticsTrain,
    TransformsTrain,
)

logger = logging.getLogger(__name__)


class MultiTable:
    """
    Relational data support for the Trainer SDK

    Args:
        relational_data (RelationalData): Core data structure representing the source tables and their relationships.
        strategy (str, optional): The strategy to use. Supports "independent" (default) and "ancestral".
        gretel_model (str, optional): The underlying Gretel model to use. Default and acceptable models vary based on strategy.
        project_display_name (str, optional): Display name in the console for a new Gretel project holding models and artifacts. Defaults to "multi-table".
        refresh_interval (int, optional): Frequency in seconds to poll Gretel Cloud for job statuses. Must be at least 30. Defaults to 60 (1m).
        backup (Backup, optional): Should not be supplied manually; instead use the `restore` classmethod.
    """

    def __init__(
        self,
        relational_data: RelationalData,
        *,
        strategy: str = "independent",
        gretel_model: Optional[str] = None,
        project_display_name: Optional[str] = None,
        refresh_interval: Optional[int] = None,
        backup: Optional[Backup] = None,
    ):
        self._strategy = _validate_strategy(strategy)
        model_name, model_config = self._validate_gretel_model(gretel_model)
        self._gretel_model = model_name
        self._model_config = model_config
        self._set_refresh_interval(refresh_interval)

        self.relational_data = relational_data
        self._artifact_collection = ArtifactCollection()
        self._latest_backup: Optional[Backup] = None
        self._transforms_train = TransformsTrain()
        self.transform_output_tables: Dict[str, pd.DataFrame] = {}
        self._synthetics_train = SyntheticsTrain()
        self._synthetics_run: Optional[SyntheticsRun] = None
        self.synthetic_output_tables: Dict[str, pd.DataFrame] = {}
        self.evaluations = defaultdict(lambda: TableEvaluation())

        configure_session(api_key="prompt", cache="yes", validate=True)

        if backup is None:
            self._complete_fresh_init(project_display_name)
        else:
            self._complete_init_from_backup(backup)

    def _complete_fresh_init(self, project_display_name: Optional[str]) -> None:
        project_display_name = project_display_name or "multi-table"
        self._project = create_project(display_name=project_display_name)
        logger.info(
            f"Created project `{self._project.display_name}` with unique name `{self._project.name}`."
        )
        self._working_dir = _mkdir(self._project.name)
        self._create_debug_summary()
        logger.info("Uploading initial configuration state to project.")
        self._upload_sources_to_project()

    def _complete_init_from_backup(self, backup: Backup) -> None:
        # Raises GretelProjectEror if not found
        self._project = get_project(name=backup.project_name)
        logger.info(
            f"Connected to existing project `{self._project.display_name}` with unique name `{self._project.name}`."
        )
        self._working_dir = _mkdir(backup.working_dir)
        self._artifact_collection = backup.artifact_collection

        # RelationalData
        source_archive_id = backup.artifact_collection.source_archive
        if source_archive_id is None:
            raise MultiTableException(
                "Cannot restore from backup: source archive is missing."
            )
        source_archive_path = self._working_dir / "source_tables.tar.gz"
        download_tar_artifact(
            self._project,
            source_archive_id,
            source_archive_path,
        )
        with tarfile.open(source_archive_path, "r:gz") as tar:
            tar.extractall(path=self._working_dir)
        for table_name, table_backup in backup.relational_data.tables.items():
            source_data = pd.read_csv(self._working_dir / f"source_{table_name}.csv")
            self.relational_data.add_table(
                name=table_name, primary_key=table_backup.primary_key, data=source_data
            )
        for fk_backup in backup.relational_data.foreign_keys:
            self.relational_data.add_foreign_key(
                foreign_key=fk_backup.foreign_key, referencing=fk_backup.referencing
            )

        # Debug summary
        debug_summary_id = backup.artifact_collection.gretel_debug_summary
        if debug_summary_id is not None:
            download_file_artifact(
                self._project,
                debug_summary_id,
                self._working_dir / "_gretel_debug_summary.json",
            )

        # Transforms Train
        backup_transforms_train = backup.transforms_train
        if backup_transforms_train is None:
            logger.info("No transforms training data found in backup.")
        else:
            logger.info("Restoring transforms models")
            self._transforms_train.models = {
                table: self._project.get_model(model_id)
                for table, model_id in backup_transforms_train.model_ids.items()
            }

        # Synthetics Train
        backup_synthetics_train = backup.synthetics_train
        if backup_synthetics_train is None:
            logger.info(
                "No synthetics training data found in backup. From here, your next step is to call `train`."
            )
            return None

        logger.info("Restoring synthetics models")

        synthetics_training_archive_id = (
            self._artifact_collection.synthetics_training_archive
        )
        if synthetics_training_archive_id is not None:
            synthetics_training_archive_path = (
                self._working_dir / "synthetics_training.tar.gz"
            )
            download_tar_artifact(
                self._project,
                synthetics_training_archive_id,
                synthetics_training_archive_path,
            )
            with tarfile.open(synthetics_training_archive_path, "r:gz") as tar:
                tar.extractall(path=self._working_dir)

        self._synthetics_train.training_columns = (
            backup_synthetics_train.training_columns
        )
        self._synthetics_train.lost_contact = backup_synthetics_train.lost_contact
        self._synthetics_train.models = {
            table: self._project.get_model(model_id)
            for table, model_id in backup_synthetics_train.model_ids.items()
        }

        still_in_progress = [
            table
            for table, model in self._synthetics_train.models.items()
            if model.status in ACTIVE_STATES
        ]
        if len(still_in_progress) > 0:
            logger.warning(
                f"Training still in progress for tables `{still_in_progress}`. From here, your next step is to wait for training to finish, and re-attempt restoring from backup once all models have completed training. You can view training progress in the Console in the `{self._project.display_name} ({self._project.name})` project."
            )
            raise MultiTableException(
                "Cannot restore while model training is actively in progress."
            )

        training_succeeded = [
            table
            for table, model in self._synthetics_train.models.items()
            if model.status == Status.COMPLETED
        ]
        for table in training_succeeded:
            model = self._synthetics_train.models[table]
            self._strategy.update_evaluation_from_model(
                table, self.evaluations, model, self._working_dir
            )

        training_failed = [
            table
            for table, model in self._synthetics_train.models.items()
            if model.status in END_STATES and table not in training_succeeded
        ]
        if len(training_failed) > 0:
            logger.info(
                f"Training failed for tables: {training_failed}. From here, your next step is to try retraining them with modified data by calling `retrain_tables`."
            )
            return None

        # Synthetics Generate
        ## First, download the outputs archive if present and extract the data.
        synthetics_outputs_archive_id = (
            self._artifact_collection.synthetics_outputs_archive
        )
        any_outputs = False
        if synthetics_outputs_archive_id is not None:
            any_outputs = True
            synthetics_output_archive_path = (
                self._working_dir / "synthetics_outputs.tar.gz"
            )
            download_tar_artifact(
                self._project,
                synthetics_outputs_archive_id,
                synthetics_output_archive_path,
            )
            with tarfile.open(synthetics_output_archive_path, "r:gz") as tar:
                tar.extractall(path=self._working_dir)

        ## Then, restore latest, potentially in-progress run data if present
        backup_generate = backup.generate
        if backup_generate is None:
            if any_outputs:
                # We shouldn't ever encounter this branch in the wild, but we define some guidance to log just in case.
                msg = "Backup included synthetics outputs archive but no latest run detail. Review previous runs' outputs (see CSVs and reports in the local directory), or start a new one by calling `generate`."
            else:
                # This branch is definitely possible / more likely.
                msg = "No generation jobs had been started in previous instance. From here, your next step is to call `generate`."
            logger.info(msg)
            return None

        record_handlers = {
            table: self._synthetics_train.models[table].get_record_handler(rh_id)
            for table, rh_id in backup_generate.record_handler_ids.items()
        }
        self._synthetics_run = SyntheticsRun(
            identifier=backup_generate.identifier,
            record_size_ratio=backup_generate.record_size_ratio,
            preserved=backup_generate.preserved,
            missing_model=backup_generate.missing_model,
            lost_contact=backup_generate.lost_contact,
            record_handlers=record_handlers,
        )

        latest_run_id = self._synthetics_run.identifier
        if latest_run_id in os.listdir(self._working_dir):
            # Latest backup was taken at a stable point (nothing actively in progress).
            for table in self.relational_data.list_all_tables():
                try:
                    self.synthetic_output_tables[table] = pd.read_csv(
                        self._working_dir / latest_run_id / f"synth_{table}.csv"
                    )
                except FileNotFoundError:
                    logger.info(
                        f"Could not find synthetic CSV for table `{table}` in run outputs."
                    )

                try:
                    self._attach_existing_reports(latest_run_id, table)
                except FileNotFoundError:
                    logger.info(
                        f"Could not find report data for table `{table}` in run outputs."
                    )
            logger.info(
                f"All tasks for generation run `{latest_run_id}` finished prior to backup. From here, you can access your synthetic data as Pandas DataFrames via `synthetic_output_tables`, or review them in CSV format along with the relational report in the local working directory."
            )
            return None
        else:
            # Latest run was still in progress. Download any seeds we may have previously created.
            for table, rh in record_handlers.items():
                data_source = rh.data_source
                if data_source is not None:
                    try:
                        download_file_artifact(
                            self._project,
                            data_source,
                            self._working_dir
                            / backup_generate.identifier
                            / f"synthetics_seed_{table}.csv",
                        )
                    except:
                        logger.warning(
                            f"Could not download seed CSV data source for `{table}`. It may have already been deleted."
                        )
            logger.info(
                f"At time of last backup, generation run `{latest_run_id}` was still in progress. From here, you can attempt to resume that generate job via `generate(resume=True)`, or restart generation from scratch via a regular call to `generate`."
            )
            return None

    @classmethod
    def restore(cls, backup_file: str) -> MultiTable:
        logger.info(f"Restoring from backup file `{backup_file}`.")
        with open(backup_file, "r") as b:
            backup = Backup.from_dict(json.load(b))

        return MultiTable(
            relational_data=RelationalData(),
            strategy=backup.strategy,
            gretel_model=backup.gretel_model,
            refresh_interval=backup.refresh_interval,
            backup=backup,
        )

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
        backup = Backup(
            project_name=self._project.name,
            strategy=self._strategy.name,
            gretel_model=self._gretel_model,
            working_dir=str(self._working_dir),
            refresh_interval=self._refresh_interval,
            artifact_collection=replace(self._artifact_collection),
            relational_data=BackupRelationalData.from_relational_data(
                self.relational_data
            ),
        )

        # Transforms Train
        if len(self._transforms_train.models) > 0:
            backup.transforms_train = BackupTransformsTrain(
                model_ids={
                    table: model.model_id
                    for table, model in self._transforms_train.models.items()
                },
                lost_contact=self._transforms_train.lost_contact,
            )

        # Synthetics Train
        if len(self._synthetics_train.models) > 0:
            backup.synthetics_train = BackupSyntheticsTrain(
                model_ids={
                    table: model.model_id
                    for table, model in self._synthetics_train.models.items()
                },
                lost_contact=self._synthetics_train.lost_contact,
                training_columns=self._synthetics_train.training_columns,
            )

        # Generate
        if self._synthetics_run is not None:
            backup.generate = BackupGenerate(
                identifier=self._synthetics_run.identifier,
                record_size_ratio=self._synthetics_run.record_size_ratio,
                preserved=self._synthetics_run.preserved,
                missing_model=self._synthetics_run.missing_model,
                lost_contact=self._synthetics_run.lost_contact,
                record_handler_ids={
                    table: rh.record_id
                    for table, rh in self._synthetics_run.record_handlers.items()
                },
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

    def train_transform_models(self, configs: Dict[str, GretelModelConfig]) -> None:
        for table, config in configs.items():
            transform_config = make_transform_config(
                self.relational_data, table, config
            )

            # Ensure consistent, friendly data source names in Console
            table_data = self.relational_data.get_table_data(table)
            transforms_train_path = self._working_dir / f"transforms_train_{table}.csv"
            table_data.to_csv(transforms_train_path, index=False)

            # Create model
            model = self._project.create_model_obj(
                model_config=transform_config, data_source=str(transforms_train_path)
            )
            self._transforms_train.models[table] = model

        self._backup()

        task = TransformsTrainTask(
            transforms_train=self._transforms_train,
            multitable=self,
        )
        run_task(task)

    def run_transforms(
        self,
        identifier: Optional[str] = None,
        in_place: bool = False,
        data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> None:
        """
        identifier: (str, optional): Unique string identifying a specific call to this method. Defaults to "transforms_" + current timestamp

        If `in_place` set to True, overwrites source data in all locations
        (internal Python state, local working directory, project artifact archive).
        Used for transforms->synthetics workflows.

        If `data` is supplied, runs only the supplied data through the corresponding transforms models.
        Otherwise runs source data through all existing transforms models.
        """
        if data is not None:
            unrunnable_tables = [
                table
                for table in data
                if not _table_trained_successfully(self._transforms_train, table)
            ]
            if len(unrunnable_tables) > 0:
                raise MultiTableException(
                    f"Cannot run transforms on provided data without successfully trained models for {unrunnable_tables}"
                )

        identifier = identifier or f"transforms_{_timestamp()}"
        logger.info(f"Starting transforms run `{identifier}`")
        run_dir = _mkdir(str(self._working_dir / identifier))
        transforms_run_paths = {}

        data = data or {
            table: self.relational_data.get_table_data(table)
            for table in self._transforms_train.models
            if _table_trained_successfully(self._transforms_train, table)
        }

        for table, df in data.items():
            transforms_run_path = run_dir / f"transforms_input_{table}.csv"
            df.to_csv(transforms_run_path, index=False)
            transforms_run_paths[table] = transforms_run_path

        transforms_record_handlers: Dict[str, RecordHandler] = {}

        for table_name, transforms_run_path in transforms_run_paths.items():
            model = self._transforms_train.models[table_name]
            record_handler = model.create_record_handler_obj(
                data_source=str(transforms_run_path)
            )
            transforms_record_handlers[table_name] = record_handler

        task = TransformsRunTask(
            record_handlers=transforms_record_handlers,
            multitable=self,
        )
        run_task(task)

        output_tables = self._strategy.label_encode_keys(
            self.relational_data, task.output_tables
        )

        if in_place:
            for table_name, transformed_table in output_tables.items():
                self.relational_data.update_table_data(table_name, transformed_table)
            self._upload_sources_to_project()

        for table, df in output_tables.items():
            filename = f"transformed_{table}.csv"
            out_path = run_dir / filename
            df.to_csv(out_path, index=False)

        archive_path = self._working_dir / "transforms_outputs.tar.gz"
        add_to_tar(archive_path, run_dir, identifier)

        self._artifact_collection.upload_transforms_outputs_archive(
            self._project, str(archive_path)
        )
        self._backup()
        self.transform_output_tables = output_tables

    def _prepare_training_data(self, tables: List[str]) -> Dict[str, Path]:
        """
        Exports a copy of each table prepared for training by the configured strategy
        to the working directory. Returns a dict with table names as keys and Paths
        to the CSVs as values.
        """
        training_data = self._strategy.prepare_training_data(self.relational_data)
        for table, df in training_data.items():
            self._synthetics_train.training_columns[table] = list(df.columns)
        training_paths = {}

        for table_name in tables:
            training_path = self._working_dir / f"synthetics_train_{table_name}.csv"
            training_data[table_name].to_csv(training_path, index=False)
            training_paths[table_name] = training_path

        return training_paths

    def _train_synthetics_models(self, training_data: Dict[str, Path]) -> None:
        for table_name, training_csv in training_data.items():
            synthetics_config = make_synthetics_config(table_name, self._model_config)
            model = self._project.create_model_obj(
                model_config=synthetics_config, data_source=str(training_csv)
            )
            self._synthetics_train.models[table_name] = model

        self._backup()

        task = SyntheticsTrainTask(
            synthetics_train=self._synthetics_train,
            multitable=self,
        )
        run_task(task)

        for table in task.completed:
            model = self._synthetics_train.models[table]
            self._strategy.update_evaluation_from_model(
                table, self.evaluations, model, self._working_dir
            )

        # TODO: consider moving this to before running the task
        archive_path = self._working_dir / "synthetics_training.tar.gz"
        for table_name, csv_path in training_data.items():
            add_to_tar(archive_path, csv_path, csv_path.name)
        self._artifact_collection.upload_synthetics_training_archive(
            self._project, str(archive_path)
        )

    def train(self) -> None:
        """Train synthetic data models on each table in the relational dataset"""
        tables = self.relational_data.list_all_tables()
        self._synthetics_train = SyntheticsTrain()

        training_data = self._prepare_training_data(tables)
        self._train_synthetics_models(training_data)

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

        for table in tables_to_retrain:
            with suppress(KeyError):
                del self._synthetics_train.models[table]
                del self._synthetics_train.training_columns[table]
        training_data = self._prepare_training_data(tables_to_retrain)
        self._train_synthetics_models(training_data)

    def _upload_sources_to_project(self) -> None:
        archive_path = self._working_dir / "source_tables.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            for table in self.relational_data.list_all_tables():
                filename = f"source_{table}.csv"
                out_path = self._working_dir / filename
                df = self.relational_data.get_table_data(table)
                df.to_csv(out_path, index=False)
                tar.add(out_path, arcname=filename)
        self._artifact_collection.upload_source_archive(
            self._project, str(archive_path)
        )
        self._backup()

    def generate(
        self,
        record_size_ratio: float = 1.0,
        preserve_tables: Optional[List[str]] = None,
        identifier: Optional[str] = None,
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
            identifier (str, optional): Unique string identifying a specific call to this method. Defaults to "synthetics_" + current timestamp.
            resume (bool, optional): Set to True when restoring from a backup to complete a previous, interrupted run.

        Returns:
            dict[str, pd.DataFrame]: Return a dictionary of table names and output data.
        """
        if resume:
            if identifier is not None:
                logger.warning(
                    "Cannot set identifier when resuming previous generation. Ignoring."
                )
            if record_size_ratio is not None:
                logger.warning(
                    "Cannot set record_size_ratio when resuming previous generation. Ignoring."
                )
            if preserve_tables is not None:
                logger.warning(
                    "Cannot set preserve_tables when resuming previous generation. Ignoring."
                )
            if self._synthetics_run is None:
                raise MultiTableException(
                    "Cannot resume a synthetics generation run without existing run information."
                )
            logger.info(f"Resuming synthetics run `{self._synthetics_run.identifier}`")
        else:
            preserve_tables = preserve_tables or []
            self._strategy.validate_preserved_tables(
                preserve_tables, self.relational_data
            )

            identifier = identifier or f"synthetics_{_timestamp()}"
            missing_model = self._list_tables_with_missing_models()

            self._synthetics_run = SyntheticsRun(
                identifier=identifier,
                record_size_ratio=record_size_ratio,
                preserved=preserve_tables,
                missing_model=missing_model,
                record_handlers={},
                lost_contact=[],
            )
            logger.info(f"Starting synthetics run `{self._synthetics_run.identifier}`")

        run_dir = _mkdir(str(self._working_dir / self._synthetics_run.identifier))

        task = SyntheticsRunTask(
            synthetics_run=self._synthetics_run,
            synthetics_train=self._synthetics_train,
            run_dir=run_dir,
            multitable=self,
        )
        run_task(task)

        output_tables = self._strategy.post_process_synthetic_results(
            task.output_tables, self._synthetics_run.preserved, self.relational_data
        )

        for table, synth_df in output_tables.items():
            synth_csv_path = run_dir / f"synth_{table}.csv"
            synth_df.to_csv(synth_csv_path, index=False)

        evaluate_project = create_project(
            display_name=f"evaluate-{self._project.display_name}"
        )
        evaluate_models = {}
        for table, synth_df in output_tables.items():
            if table in self._synthetics_run.preserved:
                continue

            evaluate_data = self._strategy.get_evaluate_model_data(
                rel_data=self.relational_data,
                table_name=table,
                synthetic_tables=output_tables,
            )
            if evaluate_data is None:
                continue

            evaluate_models[table] = evaluate_project.create_model_obj(
                model_config=make_evaluate_config(table),
                data_source=evaluate_data["synthetic"],
                ref_data=evaluate_data["source"],
            )

        synthetics_evaluate_task = SyntheticsEvaluateTask(
            evaluate_models=evaluate_models,
            project=evaluate_project,
            multitable=self,
        )
        run_task(synthetics_evaluate_task)

        for table in synthetics_evaluate_task.completed:
            self._strategy.update_evaluation_from_evaluate(
                table_name=table,
                evaluate_model=evaluate_models[table],
                evaluations=self.evaluations,
                working_dir=self._working_dir,
            )

        evaluate_project.delete()

        for table_name in output_tables:
            for eval_type in ["individual", "cross_table"]:
                for ext in ["html", "json"]:
                    filename = f"synthetics_{eval_type}_evaluation_{table_name}.{ext}"
                    with suppress(FileNotFoundError):
                        shutil.copyfile(
                            src=self._working_dir / filename,
                            dst=run_dir / filename,
                        )

        logger.info("Creating relational report")
        self.create_relational_report(
            run_identifier=self._synthetics_run.identifier,
            target_dir=run_dir,
        )

        archive_path = self._working_dir / f"synthetics_outputs.tar.gz"
        add_to_tar(archive_path, run_dir, self._synthetics_run.identifier)

        self._artifact_collection.upload_synthetics_outputs_archive(
            self._project, str(archive_path)
        )
        self.synthetic_output_tables = output_tables
        self._backup()

    def create_relational_report(self, run_identifier: str, target_dir: Path) -> None:
        presenter = ReportPresenter(
            rel_data=self.relational_data,
            evaluations=self.evaluations,
            now=datetime.utcnow(),
            run_identifier=run_identifier,
        )
        output_path = target_dir / "relational_report.html"
        with open(output_path, "w") as report:
            html_content = ReportRenderer().render(presenter)
            report.write(html_content)

    def _list_tables_with_missing_models(self) -> List[str]:
        missing_model = set()
        for table in self.relational_data.list_all_tables():
            if not _table_trained_successfully(self._synthetics_train, table):
                logger.info(
                    f"Skipping synthetic data generation for `{table}` because it does not have a trained model"
                )
                missing_model.add(table)
                for descendant in self.relational_data.get_descendants(table):
                    logger.info(
                        f"Skipping synthetic data generation for `{descendant}` because it depends on `{table}`"
                    )
                    missing_model.add(table)
        return list(missing_model)

    def _attach_existing_reports(self, run_id: str, table: str) -> None:
        individual_path = (
            self._working_dir
            / run_id
            / f"synthetics_individual_evaluation_{table}.json"
        )
        cross_table_path = (
            self._working_dir
            / run_id
            / f"synthetics_cross_table_evaluation_{table}.json"
        )

        individual_report_json = json.loads(smart_open.open(individual_path).read())
        cross_table_report_json = json.loads(smart_open.open(cross_table_path).read())

        self.evaluations[table].individual_report_json = individual_report_json
        self.evaluations[table].cross_table_report_json = cross_table_report_json

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


def _validate_strategy(strategy: str) -> Union[IndependentStrategy, AncestralStrategy]:
    strategy = strategy.lower()

    if strategy == "independent":
        return IndependentStrategy()
    elif strategy == "ancestral":
        return AncestralStrategy()
    else:
        msg = f"Unrecognized strategy requested: {strategy}. Supported strategies are `independent` and `ancestral`."
        logger.warning(msg)
        raise MultiTableException(msg)


def _table_trained_successfully(
    train_state: Union[TransformsTrain, SyntheticsTrain], table: str
) -> bool:
    model = train_state.models.get(table)
    if model is None:
        return False
    else:
        return model.status == Status.COMPLETED


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


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")
