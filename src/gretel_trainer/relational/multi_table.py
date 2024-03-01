"""
This module provides the "MultiTable" class to users. This allows you to
take extracted data from a database or data warehouse, and process it
with Gretel using Transforms, Classify, and Synthetics.
"""
from __future__ import annotations

import json
import logging
import shutil

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, cast, Optional, Union

import pandas as pd

import gretel_trainer.relational.ancestry as ancestry

from gretel_client.config import add_session_context, ClientConfig, RunnerMode
from gretel_client.projects import create_project, get_project, Project
from gretel_client.projects.artifact_handlers import open_artifact
from gretel_client.projects.jobs import ACTIVE_STATES, END_STATES, Status
from gretel_client.projects.records import RecordHandler
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
)
from gretel_trainer.relational.json import InventedTableMetadata, ProducerMetadata
from gretel_trainer.relational.model_config import (
    assemble_configs,
    get_model_key,
    make_classify_config,
    make_evaluate_config,
    make_synthetics_config,
    make_transform_config,
)
from gretel_trainer.relational.output_handler import OutputHandler, SDKOutputHandler
from gretel_trainer.relational.report.report import ReportPresenter, ReportRenderer
from gretel_trainer.relational.sdk_extras import ExtendedGretelSDK
from gretel_trainer.relational.strategies.ancestral import AncestralStrategy
from gretel_trainer.relational.strategies.independent import IndependentStrategy
from gretel_trainer.relational.table_evaluation import TableEvaluation
from gretel_trainer.relational.task_runner import run_task, TaskContext
from gretel_trainer.relational.tasks.classify import ClassifyTask
from gretel_trainer.relational.tasks.synthetics_evaluate import SyntheticsEvaluateTask
from gretel_trainer.relational.tasks.synthetics_run import SyntheticsRunTask
from gretel_trainer.relational.tasks.synthetics_train import SyntheticsTrainTask
from gretel_trainer.relational.tasks.transforms_run import TransformsRunTask
from gretel_trainer.relational.tasks.transforms_train import TransformsTrainTask
from gretel_trainer.relational.workflow_state import (
    Classify,
    SyntheticsRun,
    SyntheticsTrain,
    TransformsTrain,
)
from gretel_trainer.version import get_trainer_version

RELATIONAL_SESSION_METADATA = {"trainer_relational": get_trainer_version()}

logger = logging.getLogger(__name__)


class MultiTable:
    """
    Relational data support for the Trainer SDK

    Args:
        relational_data (RelationalData): Core data structure representing the source tables and their relationships.
        strategy (str, optional): The strategy to use for synthetics. Supports "independent" (default) and "ancestral".
        project_display_name (str, optional): Display name in the console for a new Gretel project holding models and artifacts. Defaults to "multi-table". Conflicts with `project`.
        project (Project, optional): Existing project to use for models and artifacts. Conflicts with `project_display_name`.
        refresh_interval (int, optional): Frequency in seconds to poll Gretel Cloud for job statuses. Must be at least 30. Defaults to 60 (1m).
        backup (Backup, optional): Should not be supplied manually; instead use the `restore` classmethod.
    """

    def __init__(
        self,
        relational_data: RelationalData,
        *,
        strategy: str = "independent",
        project_display_name: Optional[str] = None,
        project: Optional[Project] = None,
        refresh_interval: Optional[int] = None,
        backup: Optional[Backup] = None,
        output_handler: Optional[OutputHandler] = None,
        session: Optional[ClientConfig] = None,
    ):
        if project_display_name is not None and project is not None:
            raise MultiTableException(
                "Cannot set both `project_display_name` and `project`. "
                "Set `project_display_name` to create a new project with that display name, "
                "or set `project` to run in an existing project."
            )

        if len(cycles := relational_data.foreign_key_cycles) > 0:
            logger.warning(
                f"Detected cyclic foreign key relationships in schema: {cycles}. "
                "Support for cyclic table dependencies is limited. "
                "You may need to remove some foreign keys to ensure no cycles exist."
            )

        self._session = _configure_session(project, session)
        self._strategy = _validate_strategy(strategy)
        self._set_refresh_interval(refresh_interval)
        self.relational_data = relational_data
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
            self._complete_fresh_init(project_display_name, project, output_handler)
        else:
            # The current restore-from-backup implementation is hyper-specific to direct SDK usage.
            # We do not need to pass the Optional[OutputHandler] `output_handler` here because we know it
            # will be None; instead we create an SDKOutputHandler in that method and (for better or worse)
            # access its private `_working_dir` attribute (without pyright complaining about that attribute
            # not existing on the OutputHandler protocol).
            # In the future, other clients restoring state should implement their own `_complete_init_from...`
            # method using their own client-specific, not-None implementation of OutputHandler.
            self._complete_init_from_backup(backup)

    def _complete_fresh_init(
        self,
        project_display_name: Optional[str],
        project: Optional[Project],
        output_handler: Optional[OutputHandler],
    ) -> None:
        if project is None:
            self._project = create_project(
                display_name=project_display_name or "multi-table",
                session=self._session,
            )
            logger.info(
                f"Created project `{self._project.display_name}` with unique name `{self._project.name}`."
            )
        else:
            self._project = project.with_session(self._session)

        self._set_output_handler(output_handler, None)
        self._output_handler.save_debug_summary(self.relational_data.debug_summary())
        self._upload_sources_to_project()

    def _set_output_handler(
        self, output_handler: Optional[OutputHandler], source_archive: Optional[str]
    ) -> None:
        self._output_handler = output_handler or SDKOutputHandler(
            workdir=self._project.name,
            project=self._project,
            hybrid=self._hybrid,
            source_archive=source_archive,
        )

    def _complete_init_from_backup(self, backup: Backup) -> None:
        # Raises GretelProjectEror if not found
        self._project = get_project(name=backup.project_name, session=self._session)
        logger.info(
            f"Connected to existing project `{self._project.display_name}` with unique name `{self._project.name}`."
        )
        self._set_output_handler(None, backup.source_archive)
        # We currently only support restoring from backup via the SDK, so we know the concrete type of the output handler
        # (and set it here so pyright doesn't complain about us peeking in to a private attribute).
        self._output_handler = cast(SDKOutputHandler, self._output_handler)

        # RelationalData
        source_archive_path = self._output_handler.filepath_for("source_tables.tar.gz")
        source_archive_id = self._output_handler.get_source_archive()
        if source_archive_id is not None:
            self._extended_sdk.download_file_artifact(
                gretel_object=self._project,
                artifact_name=source_archive_id,
                out_path=source_archive_path,
            )
        if not Path(source_archive_path).exists():
            raise MultiTableException(
                "Cannot restore from backup: source archive is missing."
            )
        shutil.unpack_archive(
            filename=source_archive_path,
            extract_dir=self._output_handler._working_dir,
            format="gztar",
        )
        for table_name, table_backup in backup.relational_data.tables.items():
            source_data = self._output_handler.filepath_for(f"{table_name}.csv")
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

        # Classify
        backup_classify = backup.classify
        if backup_classify is None:
            logger.info("No classify data found in backup.")
        else:
            logger.info("Restoring classify models")
            self._classify.models = {
                table: self._project.get_model(model_id)
                for table, model_id in backup_classify.model_ids.items()
            }

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
            logger.info("No synthetics training data found in backup.")
            return None

        logger.info("Restoring synthetics models")

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
                f"Cannot restore at this time: model training still in progress for tables `{still_in_progress}`. "
                "Please wait for training to finish, and re-attempt restoring from backup once all models have completed training. "
                f"You can view training progress in the Console here: {self._project.get_console_url()}"
            )
            raise MultiTableException(
                "Cannot restore while model training is actively in progress. Wait for training to finish and try again."
            )

        training_failed = [
            table
            for table, model in self._synthetics_train.models.items()
            if model.status in END_STATES and model.status != Status.COMPLETED
        ]
        if len(training_failed) > 0:
            logger.info(
                f"Training failed for tables: {training_failed}. You may try retraining them with modified data by calling `retrain_tables`."
            )
            return None

        # Synthetics Generate
        backup_generate = backup.generate
        if backup_generate is None:
            logger.info(
                "No synthetic generation jobs had been started in previous instance."
            )
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

        artifact_keys = [artifact["key"] for artifact in self._project.artifacts]
        latest_run_id = self._synthetics_run.identifier
        if any(latest_run_id in artifact_key for artifact_key in artifact_keys):
            logger.info(
                f"Results of most recent run `{latest_run_id}` can be reviewed by downloading the output archive from the project; visit {self._project.get_console_url()}"
            )
        else:
            logger.info(
                f"At time of last backup, generation run `{latest_run_id}` was still in progress. "
                "You can attempt to resume that generate job via `generate(resume=True)`, or restart generation from scratch via a regular call to `generate`."
            )

    @classmethod
    def restore(
        cls,
        backup_file: str,
        session: Optional[ClientConfig] = None,
    ) -> MultiTable:
        """
        Create a `MultiTable` instance from a backup file.
        """
        logger.info(f"Restoring from backup file `{backup_file}`.")
        with open(backup_file, "r") as b:
            backup = Backup.from_dict(json.load(b))

        return MultiTable(
            relational_data=RelationalData(directory=backup.project_name),
            strategy=backup.strategy,
            refresh_interval=backup.refresh_interval,
            backup=backup,
            session=session,
        )

    def _backup(self) -> None:
        backup = self._build_backup()
        # exit early if nothing has changed since last backup
        if backup == self._latest_backup:
            return None

        self._output_handler.save_backup(backup)

        self._latest_backup = backup

    def _build_backup(self) -> Backup:
        backup = Backup(
            project_name=self._project.name,
            strategy=self._strategy.name,
            refresh_interval=self._refresh_interval,
            source_archive=self._output_handler.get_source_archive(),
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
        return self._session.default_runner == RunnerMode.HYBRID

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
            ctx=self._new_task_context(),
            output_handler=self._output_handler,
        )
        run_task(task, self._extended_sdk)

        self._output_handler.save_classify_outputs(task.result_filepaths)

    def _setup_transforms_train_state(self, configs: dict[str, dict]) -> None:
        for table, config in configs.items():
            model = self._project.create_model_obj(
                model_config=make_transform_config(self.relational_data, table, config),
                data_source=str(self.relational_data.get_table_source(table)),
            )
            self._transforms_train.models[table] = model

        self._backup()

    def transform_v2(
        self,
        config: GretelModelConfig,
        *,
        table_specific_configs: Optional[dict[str, GretelModelConfig]] = None,
        only: Optional[set[str]] = None,
        ignore: Optional[set[str]] = None,
        encode_keys: bool = False,
        in_place: bool = False,
        identifier: Optional[str] = None,
    ) -> None:
        configs = assemble_configs(
            self.relational_data, config, table_specific_configs, only, ignore
        )
        _validate_all_transform_v2_configs(configs)

        self._setup_transforms_train_state(configs)
        task = TransformsTrainTask(
            transforms_train=self._transforms_train,
            ctx=self._new_task_context(),
        )
        run_task(task, self._extended_sdk)

        output_tables = {}
        for table, model in self._transforms_train.models.items():
            if table in task.completed:
                with model.get_artifact_handle("data_preview") as data_preview:
                    output_tables[table] = pd.read_csv(data_preview)

        self._post_process_transformed_tables(
            output_tables=output_tables,
            identifier=identifier or f"transforms_{_timestamp()}",
            encode_keys=encode_keys,
            in_place=in_place,
        )

    def train_transforms(
        self,
        config: GretelModelConfig,
        *,
        table_specific_configs: Optional[dict[str, GretelModelConfig]] = None,
        only: Optional[set[str]] = None,
        ignore: Optional[set[str]] = None,
    ) -> None:
        configs = assemble_configs(
            self.relational_data, config, table_specific_configs, only, ignore
        )
        self._setup_transforms_train_state(configs)
        task = TransformsTrainTask(
            transforms_train=self._transforms_train,
            ctx=self._new_task_context(),
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
        if encode_keys and len(self.relational_data.foreign_key_cycles) > 0:
            raise MultiTableException(
                "Cannot encode keys when schema includes cyclic foreign key relationships."
            )

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
        transforms_run_paths = {}

        data_sources = data or {
            table: self.relational_data.get_table_source(table)
            for table in self._transforms_train.models
            if _table_trained_successfully(self._transforms_train, table)
        }

        for table, data_source in data_sources.items():
            transforms_run_path = self._output_handler.filepath_for(
                f"transforms_input_{table}.csv"
            )
            if isinstance(data_source, pd.DataFrame):
                data_source.to_csv(transforms_run_path, index=False)
            else:
                with open_artifact(data_source, "rb") as src, open_artifact(
                    transforms_run_path, "wb"
                ) as dest:
                    shutil.copyfileobj(src, dest)
            transforms_run_paths[table] = transforms_run_path

        transforms_record_handlers: dict[str, RecordHandler] = {}

        for table_name, transforms_run_path in transforms_run_paths.items():
            model = self._transforms_train.models[table_name]
            record_handler = model.create_record_handler_obj(
                data_source=transforms_run_path
            )
            transforms_record_handlers[table_name] = record_handler

        task = TransformsRunTask(
            record_handlers=transforms_record_handlers,
            ctx=self._new_task_context(),
        )
        run_task(task, self._extended_sdk)

        self._post_process_transformed_tables(
            output_tables=task.output_tables,
            identifier=identifier,
            encode_keys=encode_keys,
            in_place=in_place,
        )

    def _post_process_transformed_tables(
        self,
        output_tables: dict[str, pd.DataFrame],
        identifier: str,
        encode_keys: bool,
        in_place: bool,
    ) -> None:
        if encode_keys:
            output_tables = self._strategy.label_encode_keys(
                self.relational_data, output_tables
            )

        if in_place:
            for table_name, transformed_table in output_tables.items():
                self.relational_data.update_table_data(table_name, transformed_table)
            self._upload_sources_to_project()

        reshaped_tables = self.relational_data.restore(output_tables)

        run_subdir = self._output_handler.make_subdirectory(identifier)
        final_output_filepaths = {}
        for table, df in reshaped_tables.items():
            filename = f"transformed_{table}.csv"
            out_path = self._output_handler.filepath_for(filename, subdir=run_subdir)
            with open_artifact(
                out_path,
                "wb",
            ) as dest:
                df.to_csv(
                    dest,
                    index=False,
                    columns=self.relational_data.get_table_columns(table),
                )
            final_output_filepaths[table] = out_path

        self._output_handler.save_transforms_outputs(final_output_filepaths, run_subdir)

        self._backup()
        self.transform_output_tables = reshaped_tables

    def _train_synthetics_models(self, configs: dict[str, dict[str, Any]]) -> None:
        """
        Uses the configured strategy to prepare training data sources for each table,
        exported to the working directory. Creates a model for each table and submits
        it for training. Upon completion, downloads the evaluation reports for each
        table to the working directory.
        """
        training_paths = {
            table: self._output_handler.filepath_for(f"synthetics_train_{table}.csv")
            for table in configs
        }

        training_paths = self._strategy.prepare_training_data(
            self.relational_data, training_paths
        )

        for table_name, config in configs.items():
            if table_name not in training_paths:
                logger.info(f"Bypassing model training for table `{table_name}`")
                self._synthetics_train.bypass.append(table_name)
                continue

            synthetics_config = make_synthetics_config(table_name, config)
            model = self._project.create_model_obj(
                model_config=synthetics_config,
                data_source=training_paths[table_name],
            )
            self._synthetics_train.models[table_name] = model

        self._backup()

        task = SyntheticsTrainTask(
            synthetics_train=self._synthetics_train,
            ctx=self._new_task_context(),
        )
        run_task(task, self._extended_sdk)

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
        if len(self.relational_data.foreign_key_cycles) > 0:
            raise MultiTableException(
                "Cyclic foreign key relationships are not supported by relational synthetics."
            )

        if config is None:
            config = self._strategy.default_config

        configs = assemble_configs(
            self.relational_data, config, table_specific_configs, only, ignore
        )

        # validate table scope (preserved tables) against the strategy
        excluded_tables = [
            table
            for table in self.relational_data.list_all_tables()
            if table not in configs
        ]
        self._strategy.validate_preserved_tables(excluded_tables, self.relational_data)

        # validate all provided model configs are supported by the strategy
        for conf in configs.values():
            self._validate_synthetics_config(conf)

        self._train_synthetics_models(configs)

    def retrain_tables(self, tables: dict[str, UserFriendlyDataT]) -> None:
        """
        Provide updated table data and retrain. This method overwrites the table data in the
        `RelationalData` instance. It should be used when initial training fails and source data
        needs to be altered, but progress on other tables can be left as-is.
        """
        # The strategy determines the full set of tables that need to be retrained based on those provided.
        tables_to_retrain = self._strategy.tables_to_retrain(
            list(tables.keys()), self.relational_data
        )

        # Grab the configs from the about-to-be-replaced models. If any can't be found,
        # we have to abort because we don't know what model config to use with the new data.
        configs = {}
        for table in tables_to_retrain:
            if (old_model := self._synthetics_train.models.get(table)) is None:
                raise MultiTableException(
                    f"Could not find an existing model for table `{table}`. You may need to rerun all training via `train_synthetics`."
                )
            else:
                configs[table] = old_model.model_config

        # Orphan the old models
        for table in tables_to_retrain:
            del self._synthetics_train.models[table]

        # Update the source table data.
        for table_name, table_data in tables.items():
            self.relational_data.update_table_data(table_name, table_data)
        self._upload_sources_to_project()

        # Train new synthetics models for the subset of tables
        self._train_synthetics_models(configs)

    def _upload_sources_to_project(self) -> None:
        self._output_handler.save_sources(self.relational_data)
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
                    and table not in self._synthetics_train.bypass
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

        run_subdir = self._output_handler.make_subdirectory(
            self._synthetics_run.identifier
        )

        task = SyntheticsRunTask(
            synthetics_run=self._synthetics_run,
            synthetics_train=self._synthetics_train,
            subdir=run_subdir,
            output_handler=self._output_handler,
            ctx=self._new_task_context(),
            rel_data=self.relational_data,
            strategy=self._strategy,
        )
        run_task(task, self._extended_sdk)

        output_tables = self._strategy.post_process_synthetic_results(
            synth_tables=task.output_tables,
            preserved=self._synthetics_run.preserved,
            rel_data=self.relational_data,
            record_size_ratio=self._synthetics_run.record_size_ratio,
        )

        reshaped_tables = self.relational_data.restore(output_tables)

        synthetic_table_filepaths = {}
        for table, synth_df in reshaped_tables.items():
            synth_csv_path = self._output_handler.filepath_for(
                f"synth_{table}.csv", subdir=run_subdir
            )
            with open_artifact(synth_csv_path, "wb") as dest:
                synth_df.to_csv(
                    dest,
                    index=False,
                    columns=self.relational_data.get_table_columns(table),
                )
            synthetic_table_filepaths[table] = synth_csv_path

        individual_evaluate_models = {}
        cross_table_evaluate_models = {}
        for table, synth_df in output_tables.items():
            if table in self._synthetics_run.preserved:
                continue

            if table not in self._synthetics_train.models:
                continue

            if table not in self.relational_data.list_all_tables(Scope.EVALUATABLE):
                continue

            # Create an evaluate model for individual SQS
            individual_data = self._get_individual_evaluate_data(
                table=table,
                synthetic_tables=output_tables,
            )
            individual_sqs_job = self._project.create_model_obj(
                model_config=make_evaluate_config(table, "individual"),
                data_source=individual_data["synthetic"],
                ref_data=individual_data["source"],
            )
            individual_evaluate_models[table] = individual_sqs_job

            # Create an evaluate model for cross-table SQS (if we can/should)
            cross_table_data = self._get_cross_table_evaluate_data(
                table=table,
                synthetic_tables=output_tables,
            )
            if cross_table_data is not None:
                cross_table_sqs_job = self._project.create_model_obj(
                    model_config=make_evaluate_config(table, "cross_table"),
                    data_source=cross_table_data["synthetic"],
                    ref_data=cross_table_data["source"],
                )
                cross_table_evaluate_models[table] = cross_table_sqs_job

        synthetics_evaluate_task = SyntheticsEvaluateTask(
            individual_evaluate_models=individual_evaluate_models,
            cross_table_evaluate_models=cross_table_evaluate_models,
            subdir=run_subdir,
            output_handler=self._output_handler,
            evaluations=self._evaluations,
            ctx=self._new_task_context(),
        )
        run_task(synthetics_evaluate_task, self._extended_sdk)

        relational_report_filepath = None
        if self.relational_data.any_table_relationships():
            logger.info("Creating relational report")
            relational_report_filepath = self._output_handler.filepath_for(
                "relational_report.html", subdir=run_subdir
            )
            self.create_relational_report(
                run_identifier=self._synthetics_run.identifier,
                filepath=relational_report_filepath,
            )

        self._output_handler.save_synthetics_outputs(
            tables=synthetic_table_filepaths,
            table_reports=synthetics_evaluate_task.report_filepaths,
            relational_report=relational_report_filepath,
            run_subdir=run_subdir,
        )
        self.synthetic_output_tables = reshaped_tables
        self._backup()

    def _get_individual_evaluate_data(
        self, table: str, synthetic_tables: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """
        Returns a dictionary containing source and synthetic versions of a table,
        to be used in an Evaluate job.

        Removes all key columns to avoid artificially deflating the score
        (key types may not match, and key values carry no semantic meaning).
        """
        all_cols = self.relational_data.get_table_columns(table)
        key_cols = self.relational_data.get_all_key_columns(table)
        use_cols = [c for c in all_cols if c not in key_cols]

        return {
            "source": self.relational_data.get_table_data(table, usecols=use_cols),
            "synthetic": synthetic_tables[table].drop(columns=key_cols),
        }

    def _get_cross_table_evaluate_data(
        self, table: str, synthetic_tables: dict[str, pd.DataFrame]
    ) -> Optional[dict[str, pd.DataFrame]]:
        """
        Returns a dictionary containing source and synthetic versions of a table
        with ancestral data attached, to be used in an Evaluate job for cross-table SQS.

        Removes all key columns to avoid artificially deflating the score
        (key types may not match, and key values carry no semantic meaning).

        Returns None if a cross-table SQS job cannot or should not be performed.
        """
        # Exit early if table does not have parents (no need for cross-table evaluation)
        if len(self.relational_data.get_parents(table)) == 0:
            return None

        # Exit early if we can't create synthetic cross-table data
        # (e.g. parent data missing due to job failure)
        missing_ancestors = [
            ancestor
            for ancestor in self.relational_data.get_ancestors(table)
            if ancestor not in synthetic_tables
        ]
        if len(missing_ancestors) > 0:
            logger.info(
                f"Cannot run cross_table evaluations for `{table}` because no synthetic data exists for ancestor tables {missing_ancestors}."
            )
            return None

        source_data = ancestry.get_table_data_with_ancestors(
            self.relational_data, table
        )
        synthetic_data = ancestry.get_table_data_with_ancestors(
            self.relational_data, table, synthetic_tables
        )
        key_cols = ancestry.get_all_key_columns(self.relational_data, table)
        return {
            "source": source_data.drop(columns=key_cols),
            "synthetic": synthetic_data.drop(columns=key_cols),
        }

    def create_relational_report(self, run_identifier: str, filepath: str) -> None:
        presenter = ReportPresenter(
            rel_data=self.relational_data,
            evaluations=self.evaluations,
            now=datetime.utcnow(),
            run_identifier=run_identifier,
        )
        with open_artifact(filepath, "w") as report:
            html_content = ReportRenderer().render(presenter)
            report.write(html_content)

    def _new_task_context(self) -> TaskContext:
        return TaskContext(
            in_flight_jobs=0,
            refresh_interval=self._refresh_interval,
            project=self._project,
            extended_sdk=self._extended_sdk,
            backup=self._backup,
        )

    def _validate_synthetics_config(self, config_dict: dict[str, Any]) -> None:
        """
        Validates that the provided config (in dict form)
        is supported by the configured synthetics strategy
        """
        if (model_key := get_model_key(config_dict)) is None:
            raise MultiTableException("Invalid config")
        else:
            supported_models = self._strategy.supported_model_keys
            if model_key not in supported_models:
                raise MultiTableException(
                    f"Invalid gretel model requested: {model_key}. "
                    f"The selected strategy supports: {supported_models}."
                )


def _configure_session(
    project: Optional[Project], session: Optional[ClientConfig]
) -> ClientConfig:
    if session is None and project is not None:
        session = project.session

    return add_session_context(
        session=session, client_metrics=RELATIONAL_SESSION_METADATA
    )


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


def _validate_all_transform_v2_configs(configs: dict[str, dict]) -> None:
    invalid = 0
    for table, config in configs.items():
        model_key = get_model_key(config)
        if model_key is None or model_key != "transform_v2":
            logger.warning(
                f"Invalid model config for `{table}`, expected `transform_v2` model type, got `{model_key}`"
            )
            invalid += 1

    if invalid > 0:
        raise MultiTableException(f"Received {invalid} invalid model configs.")


def _table_trained_successfully(train_state: TransformsTrain, table: str) -> bool:
    model = train_state.models.get(table)
    if model is None:
        return False
    else:
        return model.status == Status.COMPLETED


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")
