"""
This module provides the "MultiTable" class to users. This allows you to
take extracted data from a database or data warehouse, and process it
with Gretel using Transforms, Classify, and Synthetics.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tarfile
import tempfile
from collections import defaultdict
from contextlib import suppress
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
import smart_open
from gretel_client.config import RunnerMode, get_session_config
from gretel_client.projects import Project, create_project, get_project
from gretel_client.projects.jobs import ACTIVE_STATES, END_STATES, Status
from gretel_client.projects.records import RecordHandler

from gretel_trainer.relational.artifacts import (
    ArtifactCollection,
    archive_items,
    archive_nested_dir,
)
from gretel_trainer.relational.backup import (
    Backup,
    BackupClassify,
    BackupGenerate,
    BackupRelationalData,
    BackupSyntheticsTrain,
    BackupTransformsTrain,
)
from gretel_trainer.relational.core import (
    GretelModelConfig,
    MultiTableException,
    RelationalData,
    Scope,
    UserFriendlyDataT,
    skip_table,
)
from gretel_trainer.relational.json import InventedTableMetadata, ProducerMetadata
from gretel_trainer.relational.log import silent_logs
from gretel_trainer.relational.model_config import (
    get_model_key,
    ingest,
    make_classify_config,
    make_evaluate_config,
    make_synthetics_config,
    make_transform_config,
)
from gretel_trainer.relational.report.report import ReportPresenter, ReportRenderer
from gretel_trainer.relational.sdk_extras import ExtendedGretelSDK
from gretel_trainer.relational.strategies.ancestral import AncestralStrategy
from gretel_trainer.relational.strategies.independent import IndependentStrategy
from gretel_trainer.relational.table_evaluation import TableEvaluation
from gretel_trainer.relational.task_runner import run_task
from gretel_trainer.relational.tasks import (
    ClassifyTask,
    SyntheticsEvaluateTask,
    SyntheticsRunTask,
    SyntheticsTrainTask,
    TransformsRunTask,
    TransformsTrainTask,
)
from gretel_trainer.relational.workflow_state import (
    Classify,
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
        strategy (str, optional): The strategy to use for synthetics. Supports "independent" (default) and "ancestral".
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
        if gretel_model is not None:
            if backup is None:
                logger.warning(
                    "The `gretel_model` argument is deprecated and will be removed in a future release. "
                    "Going forward you should provide a config to `train_synthetics`."
                )
            model_name, model_config = self._validate_gretel_model(gretel_model)
            self._gretel_model = model_name
            self._model_config = model_config
        else:
            # Set these to the original default for backwards compatibility.
            # When we completely remove the `gretel_model` init param, these attrs can be removed as well.
            # We don't need to validate here because the default model (amplify) works with both strategies.
            self._gretel_model = "amplify"
            self._model_config = "synthetics/amplify"
        self._set_refresh_interval(refresh_interval)
        self.relational_data = relational_data
        self._artifact_collection = ArtifactCollection(hybrid=self._hybrid)
        self._extended_sdk = ExtendedGretelSDK(hybrid=self._hybrid)
        self._latest_backup: Optional[Backup] = None
        self._classify = Classify()
        self._transforms_train = TransformsTrain()
        self.transform_output_tables: dict[str, pd.DataFrame] = {}
        self._synthetics_train = SyntheticsTrain()
        self._synthetics_run: Optional[SyntheticsRun] = None
        self.synthetic_output_tables: dict[str, pd.DataFrame] = {}
        self._evaluations = defaultdict(lambda: TableEvaluation())

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
        source_archive_path = self._working_dir / "source_tables.tar.gz"
        source_archive_id = backup.artifact_collection.source_archive
        if source_archive_id is not None:
            self._extended_sdk.download_tar_artifact(
                self._project,
                source_archive_id,
                source_archive_path,
            )
        if not source_archive_path.exists():
            raise MultiTableException(
                "Cannot restore from backup: source archive is missing."
            )
        with tarfile.open(source_archive_path, "r:gz") as tar:
            tar.extractall(path=self._working_dir)
        for table_name, table_backup in backup.relational_data.tables.items():
            source_data = self._working_dir / f"source_{table_name}.csv"
            invented_table_metadata = None
            producer_metadata = None
            if (imeta := table_backup.invented_table_metadata) is not None:
                invented_table_metadata = InventedTableMetadata(**imeta)
            if (pmeta := table_backup.producer_metadata) is not None:
                producer_metadata = ProducerMetadata(**pmeta)
            self.relational_data._add_single_table(
                name=table_name,
                primary_key=table_backup.primary_key,
                source=source_data,
                invented_table_metadata=invented_table_metadata,
                producer_metadata=producer_metadata,
            )
        for fk_backup in backup.relational_data.foreign_keys:
            self.relational_data.add_foreign_key_constraint(
                table=fk_backup.table,
                constrained_columns=fk_backup.constrained_columns,
                referred_table=fk_backup.referred_table,
                referred_columns=fk_backup.referred_columns,
            )

        # Debug summary
        debug_summary_id = backup.artifact_collection.gretel_debug_summary
        if debug_summary_id is not None:
            self._extended_sdk.download_file_artifact(
                self._project,
                debug_summary_id,
                self._working_dir / "_gretel_debug_summary.json",
            )

        # Classify
        ## First, download the outputs archive if present and extract the data.
        classify_outputs_archive_path = self._working_dir / "classify_outputs.tar.gz"
        if (
            classify_outputs_archive_id := backup.artifact_collection.classify_outputs_archive
        ) is not None:
            self._extended_sdk.download_tar_artifact(
                self._project,
                classify_outputs_archive_id,
                classify_outputs_archive_path,
            )
        if classify_outputs_archive_path.exists():
            with tarfile.open(classify_outputs_archive_path, "r:gz") as tar:
                tar.extractall(path=self._working_dir)

        ## Then, restore model state if present
        backup_classify = backup.classify
        if backup_classify is None:
            logger.info("No classify data found in backup.")
        else:
            logger.info("Restoring classify models")
            self._classify.models = {
                table: self._project.get_model(model_id)
                for table, model_id in backup_classify.model_ids.items()
            }
            ...

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

        synthetics_training_archive_path = (
            self._working_dir / "synthetics_training.tar.gz"
        )
        synthetics_training_archive_id = (
            self._artifact_collection.synthetics_training_archive
        )
        if synthetics_training_archive_id is not None:
            self._extended_sdk.download_tar_artifact(
                self._project,
                synthetics_training_archive_id,
                synthetics_training_archive_path,
            )
        if synthetics_training_archive_path.exists():
            with tarfile.open(synthetics_training_archive_path, "r:gz") as tar:
                tar.extractall(path=self._working_dir)

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
            if table in self.relational_data.list_all_tables(Scope.EVALUATABLE):
                model = self._synthetics_train.models[table]
                with silent_logs():
                    self._strategy.update_evaluation_from_model(
                        table,
                        self._evaluations,
                        model,
                        self._working_dir,
                        self._extended_sdk,
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
        synthetics_output_archive_path = self._working_dir / "synthetics_outputs.tar.gz"
        synthetics_outputs_archive_id = (
            self._artifact_collection.synthetics_outputs_archive
        )
        any_outputs = False
        if synthetics_outputs_archive_id is not None:
            self._extended_sdk.download_tar_artifact(
                self._project,
                synthetics_outputs_archive_id,
                synthetics_output_archive_path,
            )
        if synthetics_output_archive_path.exists():
            any_outputs = True
            # Extract the nested archives to a temporary directory, and then
            # extract the contents of each run archive to a subdir in the working directory
            with tarfile.open(
                synthetics_output_archive_path, "r:gz"
            ) as tar, tempfile.TemporaryDirectory() as tmpdir:
                tar.extractall(path=tmpdir)
                for run_tar in os.listdir(tmpdir):
                    with tarfile.open(f"{tmpdir}/{run_tar}", "r:gz") as rt:
                        rt.extractall(
                            path=self._working_dir / run_tar.removesuffix(".tar.gz")
                        )

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
                f"All tasks for generation run `{latest_run_id}` finished prior to backup. From here, you can review your synthetic data in CSV format along with the relational report in the local working directory."
            )
            return None
        else:
            # Latest run was still in progress. Download any seeds we may have previously created.
            for table, rh in record_handlers.items():
                data_source = rh.data_source
                if data_source is not None:
                    try:
                        self._extended_sdk.download_file_artifact(
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
        """
        Create a `MultiTable` instance from a backup file.
        """
        logger.info(f"Restoring from backup file `{backup_file}`.")
        with open(backup_file, "r") as b:
            backup = Backup.from_dict(json.load(b))

        return MultiTable(
            relational_data=RelationalData(directory=backup.working_dir),
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

        _upload_gretel_backup(self._project, str(backup_path), self._hybrid)

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

        # Classify
        if len(self._classify.models) > 0:
            backup.classify = BackupClassify(
                model_ids={
                    table: model.model_id
                    for table, model in self._classify.models.items()
                }
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
            )

        # Generate
        if self._synthetics_run is not None:
            backup.generate = BackupGenerate(
                identifier=self._synthetics_run.identifier,
                record_size_ratio=self._synthetics_run.record_size_ratio,
                preserved=self._synthetics_run.preserved,
                lost_contact=self._synthetics_run.lost_contact,
                record_handler_ids={
                    table: rh.record_id
                    for table, rh in self._synthetics_run.record_handlers.items()
                },
            )

        return backup

    @property
    def _hybrid(self) -> bool:
        return get_session_config().default_runner == RunnerMode.HYBRID

    @property
    def evaluations(self) -> dict[str, TableEvaluation]:
        evaluations = defaultdict(lambda: TableEvaluation())

        for table, evaluation in self._evaluations.items():
            if (public_name := self.relational_data.get_public_name(table)) is not None:
                evaluations[public_name] = evaluation

        return evaluations

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

    def classify(self, config: GretelModelConfig, all_rows: bool = False) -> None:
        classify_data_sources = {}
        for table in self.relational_data.list_all_tables():
            classify_config = make_classify_config(table, config)
            data_source = str(self.relational_data.get_table_source(table))
            classify_data_sources[table] = data_source

            # Create model if necessary
            if self._classify.models.get(table) is not None:
                continue

            model = self._project.create_model_obj(
                model_config=classify_config, data_source=data_source
            )
            self._classify.models[table] = model

        self._backup()

        task = ClassifyTask(
            classify=self._classify,
            data_sources=classify_data_sources,
            all_rows=all_rows,
            multitable=self,
            out_dir=self._working_dir,
        )
        run_task(task, self._extended_sdk)

        archive_path = self._working_dir / "classify_outputs.tar.gz"
        archive_items(archive_path, task.result_filepaths)
        self._artifact_collection.upload_classify_outputs_archive(
            self._project, str(archive_path)
        )

    def _setup_transforms_train_state(
        self, configs: dict[str, GretelModelConfig]
    ) -> None:
        for table, config in configs.items():
            model = self._project.create_model_obj(
                model_config=make_transform_config(self.relational_data, table, config),
                data_source=str(self.relational_data.get_table_source(table)),
            )
            self._transforms_train.models[table] = model

        self._backup()

    def train_transform_models(self, configs: dict[str, GretelModelConfig]) -> None:
        """
        DEPRECATED: Please use `train_transforms` instead.
        """
        logger.warning(
            "This method is deprecated and will be removed in a future release. "
            "Please use `train_transforms` instead."
        )
        use_configs = {}
        for table, config in configs.items():
            for m_table in self.relational_data.get_modelable_table_names(table):
                use_configs[m_table] = config

        self._setup_transforms_train_state(use_configs)
        task = TransformsTrainTask(
            transforms_train=self._transforms_train,
            multitable=self,
        )
        run_task(task, self._extended_sdk)

    def train_transforms(
        self,
        config: GretelModelConfig,
        *,
        only: Optional[set[str]] = None,
        ignore: Optional[set[str]] = None,
    ) -> None:
        only, ignore = self._get_only_and_ignore(only, ignore)

        configs = {
            table: config
            for table in self.relational_data.list_all_tables()
            if not skip_table(table, only, ignore)
        }

        self._setup_transforms_train_state(configs)
        task = TransformsTrainTask(
            transforms_train=self._transforms_train,
            multitable=self,
        )
        run_task(task, self._extended_sdk)

    def run_transforms(
        self,
        identifier: Optional[str] = None,
        in_place: bool = False,
        data: Optional[dict[str, UserFriendlyDataT]] = None,
        encode_keys: bool = False,
    ) -> None:
        """
        Run pre-trained Gretel Transform models on Relational table data:

        Args:
            identifier: Unique string identifying a specific call to this method. Defaults to `transforms_` + current timestamp
            in_place: If True, overwrites source data in all locations
                (internal Python state, local working directory, project artifact archive).
                Used for transforms->synthetics workflows.
            data: If supplied, runs only the supplied data through the corresponding transforms models.
                Otherwise runs source data through all existing transforms models.
            encode_keys: If set, primary and foreign keys will be replaced with label encoded variants. This can add
                an additional level of privacy at the cost of referential integrity between transformed and
                original data.
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

        data_sources = data or {
            table: str(self.relational_data.get_table_source(table))
            for table in self._transforms_train.models
            if _table_trained_successfully(self._transforms_train, table)
        }

        for table, data_source in data_sources.items():
            transforms_run_path = run_dir / f"transforms_input_{table}.csv"
            if isinstance(data_source, pd.DataFrame):
                data_source.to_csv(transforms_run_path, index=False)
            else:
                shutil.copyfile(data_source, transforms_run_path)
            transforms_run_paths[table] = transforms_run_path

        transforms_record_handlers: dict[str, RecordHandler] = {}

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
        run_task(task, self._extended_sdk)

        output_tables = task.output_tables
        if encode_keys:
            output_tables = self._strategy.label_encode_keys(
                self.relational_data, task.output_tables
            )

        if in_place:
            for table_name, transformed_table in output_tables.items():
                self.relational_data.update_table_data(table_name, transformed_table)
            self._upload_sources_to_project()

        reshaped_tables = self.relational_data.restore(output_tables)

        for table, df in reshaped_tables.items():
            filename = f"transformed_{table}.csv"
            out_path = run_dir / filename
            df.to_csv(
                out_path,
                index=False,
                columns=self.relational_data.get_table_columns(table),
            )

        all_runs_archive_path = self._working_dir / "transforms_outputs.tar.gz"

        archive_nested_dir(
            targz=all_runs_archive_path,
            directory=run_dir,
            name=identifier,
        )

        self._artifact_collection.upload_transforms_outputs_archive(
            self._project, str(all_runs_archive_path)
        )
        self._backup()
        self.transform_output_tables = reshaped_tables

    def _get_only_and_ignore(
        self, only: Optional[set[str]], ignore: Optional[set[str]]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """
        Accepts the `only` and `ignore` parameter values as provided by the user and:
        - ensures both are not set (must provide one or the other, or neither)
        - translates any JSON-source tables to the invented tables
        """
        if only is not None and ignore is not None:
            raise MultiTableException("Cannot specify both `only` and `ignore`.")

        modelable_tables = set()
        for table in only or ignore or {}:
            m_names = self.relational_data.get_modelable_table_names(table)
            if len(m_names) == 0:
                raise MultiTableException(f"Unrecognized table name: `{table}`")
            modelable_tables.update(m_names)

        if only is None:
            return (None, modelable_tables)
        elif ignore is None:
            return (modelable_tables, None)
        else:
            return (None, None)

    def _train_synthetics_models(self, configs: dict[str, dict[str, Any]]) -> None:
        """
        Uses the configured strategy to prepare training data sources for each table,
        exported to the working directory. Creates a model for each table and submits
        it for training. Upon completion, downloads the evaluation reports for each
        table to the working directory.
        """
        training_paths = {
            table: self._working_dir / f"synthetics_train_{table}.csv"
            for table in configs
        }

        self._strategy.prepare_training_data(self.relational_data, training_paths)

        for table_name, config in configs.items():
            synthetics_config = make_synthetics_config(table_name, config)
            model = self._project.create_model_obj(
                model_config=synthetics_config,
                data_source=str(training_paths[table_name]),
            )
            self._synthetics_train.models[table_name] = model

        archive_path = self._working_dir / "synthetics_training.tar.gz"
        archive_items(archive_path, list(training_paths.values()))
        self._artifact_collection.upload_synthetics_training_archive(
            self._project, str(archive_path)
        )

        self._backup()

        task = SyntheticsTrainTask(
            synthetics_train=self._synthetics_train,
            multitable=self,
        )
        run_task(task, self._extended_sdk)

        for table in task.completed:
            if table in self.relational_data.list_all_tables(Scope.EVALUATABLE):
                model = self._synthetics_train.models[table]
                self._strategy.update_evaluation_from_model(
                    table,
                    self._evaluations,
                    model,
                    self._working_dir,
                    self._extended_sdk,
                )

    def train(self) -> None:
        """
        DEPRECATED: Please use `train_synthetics` instead.
        """
        logger.warning(
            "This method is deprecated and will be removed in a future release. "
            "Please use `train_synthetics` instead."
        )
        # This method completely resets any existing SyntheticsTrain state,
        # orphaning any existing models in the project.
        self._synthetics_train = SyntheticsTrain()

        # This method only supports using a single config
        # (the blueprint config set at MultiTable initialization)
        # and cannot omit any tables from training.
        config = ingest(self._model_config)
        configs = {table: config for table in self.relational_data.list_all_tables()}

        self._train_synthetics_models(configs)

    def train_synthetics(
        self,
        *,
        config: Optional[GretelModelConfig] = None,
        table_specific_configs: Optional[dict[str, GretelModelConfig]] = None,
        only: Optional[set[str]] = None,
        ignore: Optional[set[str]] = None,
    ) -> None:
        """
        Train synthetic data models for the tables in the tableset,
        optionally scoped by either `only` or `ignore`.
        """
        only, ignore = self._get_only_and_ignore(only, ignore)

        include_tables: list[str] = []
        omit_tables: list[str] = []
        for table in self.relational_data.list_all_tables():
            if skip_table(table, only, ignore):
                omit_tables.append(table)
            else:
                include_tables.append(table)

        # TODO: Ancestral strategy requires that for each table omitted from synthetics ("preserved"),
        # all its ancestors must also be omitted. In the future, it'd be nice to either find a way to
        # eliminate this requirement completely, or (less ideal) allow incremental training of tables,
        # e.g. train a few in one "batch", then a few more before generating (perhaps with some logs
        # along the way informing the user of which required tables are missing).
        self._strategy.validate_preserved_tables(omit_tables, self.relational_data)

        # Translate any JSON-source tables in table_specific_configs to invented tables
        all_table_specific_configs = {}
        for table, conf in (table_specific_configs or {}).items():
            m_names = self.relational_data.get_modelable_table_names(table)
            if len(m_names) == 0:
                raise MultiTableException(f"Unrecognized table name: `{table}`")
            all_table_specific_configs.update({m: conf for m in m_names})

        # Ensure compatibility between only/ignore and table_specific_configs
        omitted_tables_with_overrides_specified = []
        for table in all_table_specific_configs:
            if table in omit_tables:
                omitted_tables_with_overrides_specified.append(table)
        if len(omitted_tables_with_overrides_specified) > 0:
            raise MultiTableException(
                f"Cannot provide configs for tables that have been omitted from synthetics training: "
                f"{omitted_tables_with_overrides_specified}"
            )

        # Validate the provided config
        # Currently an optional argument for backwards compatibility; if None, fall back to the one configured
        # on the MultiTable instance via the deprecated `gretel_model` parameter
        if config is not None:
            default_config_dict = self._validate_synthetics_config(config)
        else:
            logger.warning(
                "Calling `train_synthetics` without specifying a `config` is deprecated; "
                "in a future release, this argument will be required. "
                "For now, falling back to the model configured on the MultiTable instance "
                "(which is also deprecated and scheduled for removal)."
            )
            default_config_dict = ingest(self._model_config)

        # Validate any table-specific configs
        table_specific_config_dicts = {
            table: self._validate_synthetics_config(conf)
            for table, conf in all_table_specific_configs.items()
        }

        configs = {
            table: table_specific_config_dicts.get(table, default_config_dict)
            for table in include_tables
        }

        self._train_synthetics_models(configs)

    def retrain_tables(self, tables: dict[str, UserFriendlyDataT]) -> None:
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

        config = ingest(self._model_config)
        configs = {table: config for table in tables_to_retrain}

        self._train_synthetics_models(configs)

    def _upload_sources_to_project(self) -> None:
        archive_path = self._working_dir / "source_tables.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            for table in self.relational_data.list_all_tables(Scope.ALL):
                source_path = Path(self.relational_data.get_table_source(table))
                filename = source_path.name
                tar.add(source_path, arcname=f"source_{filename}")
        self._artifact_collection.upload_source_archive(
            self._project, str(archive_path)
        )
        self._backup()

    def generate(
        self,
        record_size_ratio: float = 1.0,
        preserve_tables: Optional[list[str]] = None,
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
            identifier (str, optional): Unique string identifying a specific call to this method. Defaults to `synthetics_` + current timestamp.
            resume (bool, optional): Set to True when restoring from a backup to complete a previous, interrupted run.
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
            preserve_tables.extend(
                [
                    table
                    for table in self.relational_data.list_all_tables()
                    if table not in self._synthetics_train.models
                ]
            )
            self._strategy.validate_preserved_tables(
                preserve_tables, self.relational_data
            )

            identifier = identifier or f"synthetics_{_timestamp()}"

            self._synthetics_run = SyntheticsRun(
                identifier=identifier,
                record_size_ratio=record_size_ratio,
                preserved=preserve_tables,
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
        run_task(task, self._extended_sdk)

        output_tables = self._strategy.post_process_synthetic_results(
            synth_tables=task.output_tables,
            preserved=self._synthetics_run.preserved,
            rel_data=self.relational_data,
            record_size_ratio=self._synthetics_run.record_size_ratio,
        )

        reshaped_tables = self.relational_data.restore(output_tables)

        for table, synth_df in reshaped_tables.items():
            synth_csv_path = run_dir / f"synth_{table}.csv"
            synth_df.to_csv(
                synth_csv_path,
                index=False,
                columns=self.relational_data.get_table_columns(table),
            )

        evaluate_project = create_project(
            display_name=f"evaluate-{self._project.display_name}"
        )
        evaluate_models = {}
        for table, synth_df in output_tables.items():
            if table in self._synthetics_run.preserved:
                continue

            if table not in self._synthetics_train.models:
                continue

            if table not in self.relational_data.list_all_tables(Scope.EVALUATABLE):
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
        run_task(synthetics_evaluate_task, self._extended_sdk)

        # Tables passed to task were already scoped to evaluatable tables
        for table in synthetics_evaluate_task.completed:
            self._strategy.update_evaluation_from_evaluate(
                table_name=table,
                evaluate_model=evaluate_models[table],
                evaluations=self._evaluations,
                working_dir=self._working_dir,
                extended_sdk=self._extended_sdk,
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

        all_runs_archive_path = self._working_dir / "synthetics_outputs.tar.gz"

        archive_nested_dir(
            targz=all_runs_archive_path,
            directory=run_dir,
            name=self._synthetics_run.identifier,
        )

        self._artifact_collection.upload_synthetics_outputs_archive(
            self._project, str(all_runs_archive_path)
        )
        self.synthetic_output_tables = reshaped_tables
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

        self._evaluations[table].individual_report_json = individual_report_json
        self._evaluations[table].cross_table_report_json = cross_table_report_json

    def _validate_gretel_model(self, gretel_model: str) -> tuple[str, str]:
        supported_models = self._strategy.supported_gretel_models
        if gretel_model not in supported_models:
            msg = f"Invalid gretel model requested: {gretel_model}. The selected strategy supports: {supported_models}."
            logger.warning(msg)
            raise MultiTableException(msg)

        _BLUEPRINTS = {
            "amplify": "synthetics/amplify",
            "actgan": "synthetics/tabular-actgan",
            "lstm": "synthetics/tabular-lstm",
            "tabular-dp": "synthetics/tabular-differential-privacy",
        }

        return (gretel_model, _BLUEPRINTS[gretel_model])

    def _validate_synthetics_config(self, config: GretelModelConfig) -> dict[str, Any]:
        """
        Validates that the provided config:
        - has the general shape of a Gretel model config (or can be read into one, e.g. blueprints)
        - is supported by the configured synthetics strategy

        Returns the parsed config as read by read_model_config.
        """
        config_dict = ingest(config)
        if (model_key := get_model_key(config_dict)) is None:
            raise MultiTableException("Invalid config")
        else:
            supported_models = self._strategy.supported_model_keys
            if model_key not in supported_models:
                raise MultiTableException(
                    f"Invalid gretel model requested: {model_key}. "
                    f"The selected strategy supports: {supported_models}."
                )

        return config_dict


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


def _table_trained_successfully(train_state: TransformsTrain, table: str) -> bool:
    model = train_state.models.get(table)
    if model is None:
        return False
    else:
        return model.status == Status.COMPLETED


def _mkdir(name: str) -> Path:
    d = Path(name)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _upload_gretel_backup(project: Project, path: str, hybrid: bool) -> None:
    if hybrid:
        return None
    latest = project.upload_artifact(path)
    for artifact in project.artifacts:
        key = artifact["key"]
        if key != latest and key.endswith("__gretel_backup.json"):
            project.delete_artifact(key)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")
