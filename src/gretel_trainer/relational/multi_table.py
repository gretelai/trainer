import json
import logging
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from gretel_client import configure_session
from gretel_client.projects import create_or_get_unique_project
from gretel_client.projects.jobs import END_STATES, Job, Status
from gretel_client.projects.models import Model, read_model_config
from gretel_client.projects.records import RecordHandler

from gretel_trainer.relational.core import (
    MultiTableException,
    RelationalData,
    TableEvaluation,
)
from gretel_trainer.relational.strategies.ancestral import AncestralStrategy
from gretel_trainer.relational.strategies.independent import IndependentStrategy

GretelModelConfig = Union[str, Path, Dict]

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


class MultiTable:
    """
    Relational data support for the Trainer SDK

    Args:
        relational_data (RelationalData): Core data structure representing the source tables and their relationships.
        project_name (str, optional): Name for the Gretel project holding models and artifacts. Defaults to "multi-table".
        strategy (str, optional): The strategy to use. Supports "independent" (default) and "ancestral".
        gretel_model (str, optional): The underlying Gretel model to use. Default and acceptable models vary based on strategy.
        working_dir (str, optional): Directory in which temporary assets should be cached. Defaults to match the project_name.
        refresh_interval (int, optional): Frequency in seconds to poll Gretel Cloud for job statuses. Must be at least 60 (1m). Defaults to 180 (3m).
    """

    def __init__(
        self,
        relational_data: RelationalData,
        project_name: str = "multi-table",
        strategy: str = "independent",
        gretel_model: Optional[str] = None,
        working_dir: Optional[str] = None,
        refresh_interval: Optional[int] = None,
    ):
        self._strategy = _validate_strategy(strategy)
        self._model_config = self._validate_gretel_model(gretel_model)

        configure_session(api_key="prompt", cache="yes", validate=True)
        self._project = create_or_get_unique_project(name=project_name)

        self.relational_data = relational_data
        self._set_refresh_interval(refresh_interval)
        self._models = {}
        self.train_statuses = {}
        self._training_columns = {}
        self._reset_train_statuses(self.relational_data.list_all_tables())
        self._reset_generation_statuses()
        self.evaluations = defaultdict(lambda: TableEvaluation())

        working_dir = working_dir or project_name
        self._working_dir = Path(working_dir)
        os.makedirs(self._working_dir, exist_ok=True)
        self._create_debug_summary()

    @property
    def state_by_action(self) -> Dict[str, Dict[str, Any]]:
        return {
            "train": self.train_statuses,
            "generate": self.generate_statuses,
        }

    @property
    def state_by_table(self) -> Dict[str, Dict[str, Any]]:
        return {
            table_name: self._table_state(table_name)
            for table_name in self.relational_data.list_all_tables()
        }

    def _table_state(self, table_name: str) -> Dict[str, Any]:
        return {
            "train": self.train_statuses[table_name],
            "generate": self.generate_statuses[table_name],
        }

    def _set_refresh_interval(self, interval: Optional[int]) -> None:
        if interval is None:
            self._refresh_interval = 180
        else:
            if interval < 60:
                logger.warning(
                    "Refresh interval must be at least 60 seconds. Setting to 60."
                )
                self._refresh_interval = 60
            else:
                self._refresh_interval = interval

    def _create_debug_summary(self) -> None:
        debug_summary_path = self._working_dir / "_gretel_debug_summary.json"
        with open(debug_summary_path, "w") as dbg:
            json.dump(self.relational_data.debug_summary(), dbg)
        self._project.upload_artifact(str(debug_summary_path))

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
            for table_name, transformed_table in output_tables:
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
                if refresh_attempts[table_name] >= 5:
                    self._log_lost_contact(table_name)
                    record_handler_statuses[table_name] = Status.LOST
                    continue

                # If RH is not finished but model training is, update RH status and handle
                if model_statuses.get(table_name) in END_STATES:
                    record_handler = record_handlers[table_name]
                    rh_status = _cautiously_refresh_status(
                        record_handler, table_name, refresh_attempts
                    )
                    record_handler_statuses[table_name] = rh_status

                    if rh_status == Status.COMPLETED:
                        self._log_success(table_name, "transforms data generation")
                        out_table = pd.read_csv(
                            record_handler.get_artifact_link("data"), compression="gzip"
                        )
                        output_tables[table_name] = out_table
                    elif rh_status in END_STATES:
                        self._log_failed(table_name, "transforms data generation")
                    else:
                        self._log_in_progress(table_name, "transforms data generation")

                    continue

                # Here = model training was last seen in progress. Update model status and handle.
                model = models[table_name]
                model_status = _cautiously_refresh_status(
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
                    self._log_in_progress(table_name, "transforms model training")

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
            training_path = self._working_dir / f"train_{table_name}.csv"
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
                if refresh_attempts[table_name] >= 5:
                    self._log_lost_contact(table_name)
                    self.train_statuses[table_name] = TrainStatus.Failed
                    continue

                model = self._models[table_name]

                status = _cautiously_refresh_status(model, table_name, refresh_attempts)

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
                    self._log_in_progress(table_name, "model training")
                    continue

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

        tables_to_retrain = self._strategy.tables_to_retrain(
            list(tables.keys()), self.relational_data
        )

        self._reset_train_statuses(tables_to_retrain)
        training_data = self._prepare_training_data(tables_to_retrain)
        self._train_models(training_data)

    def _reset_train_statuses(self, tables: List[str]) -> None:
        for table in tables:
            self.train_statuses[table] = TrainStatus.NotStarted

    def generate(
        self,
        record_size_ratio: float = 1.0,
        preserve_tables: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Sample synthetic data from trained models.
        Tables that did not train successfully will be omitted from the output dictionary.
        Tables listed in `preserve_tables` *may differ* from source tables in foreign key columns, to ensure
        joining to parent tables (which may have been synthesized) continues to work properly.

        Args:
            record_size_ratio (float, optional): Ratio to upsample real world data size with. Defaults to 1.
            preserve_tables (list[str], optional): List of tables to skip sampling and leave (mostly) identical to source.

        Returns:
            dict[str, pd.DataFrame]: Return a dictionary of table names and output data.
        """
        self._reset_generation_statuses()
        output_tables = {}

        preserve_tables = preserve_tables or []
        self._strategy.validate_preserved_tables(preserve_tables, self.relational_data)
        self._skip_some_tables(preserve_tables, output_tables)

        all_tables = self.relational_data.list_all_tables()
        record_handlers: Dict[str, RecordHandler] = {}
        refresh_attempts: Dict[str, int] = defaultdict(int)

        def _more_to_do() -> bool:
            return not all(
                [
                    self._table_generation_in_terminal_state(table)
                    for table in all_tables
                ]
            )

        while _more_to_do():
            # Don't wait needlessly the first time through.
            if len(record_handlers) > 0:
                self._wait_refresh_interval()

            for table_name, record_handler in record_handlers.items():
                # No need to do anything with tables in terminal state
                if self._table_generation_in_terminal_state(table_name):
                    continue

                # If we consistently failed to refresh the job via API, fail the table
                if refresh_attempts[table_name] >= 5:
                    self._log_lost_contact(table_name)
                    self.generate_statuses[table_name] = GenerateStatus.Failed
                    continue

                status = _cautiously_refresh_status(
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
                    self._log_in_progress(table_name, "synthetic data generation")
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
                    record_size_ratio,
                    output_tables,
                    self._working_dir,
                    self._training_columns[table_name],
                )
                self._log_start(table_name, "synthetic data generation")
                self.generate_statuses[table_name] = GenerateStatus.InProgress
                model = self._models[table_name]
                record_handler = model.create_record_handler_obj(**table_job)
                record_handler.submit_cloud()
                record_handlers[table_name] = record_handler

        output_tables = self._strategy.post_process_synthetic_results(
            output_tables, preserve_tables, self.relational_data
        )
        self.synthetic_output_tables = output_tables

        for table_name, df in self.synthetic_output_tables.items():
            out_path = self._working_dir / f"synth_{table_name}.csv"
            df.to_csv(out_path, index=False)
            self._project.upload_artifact(out_path)

        return self.synthetic_output_tables

    def expand_evaluations(self) -> None:
        """
        Adds evaluation metrics for the "opposite" correlation strategy using the Gretel Evaluate API.
        """
        for table_name in self.synthetic_output_tables:
            logger.info(
                f"Expanding evaluation metrics for `{table_name}` via Gretel Evaluate API."
            )
            self._strategy.update_evaluation_via_evaluate(
                evaluation=self.evaluations[table_name],
                table=table_name,
                rel_data=self.relational_data,
                synthetic_tables=self.synthetic_output_tables,
                working_dir=self._working_dir,
            )

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

    def _wait_refresh_interval(self) -> None:
        logger.info(f"Next status check in {self._refresh_interval} seconds.")
        time.sleep(self._refresh_interval)

    def _log_start(self, table_name: str, action: str) -> None:
        logger.info(f"Starting {action} for `{table_name}`.")

    def _log_in_progress(self, table_name: str, action: str) -> None:
        logger.info(f"{action.capitalize()} job for `{table_name}` still in progress.")

    def _log_failed(self, table_name: str, action: str) -> None:
        logger.info(f"{action.capitalize()} failed for `{table_name}`.")

    def _log_success(self, table_name: str, action: str) -> None:
        logger.info(f"{action.capitalize()} successfully completed for `{table_name}`.")

    def _log_lost_contact(self, table_name: str) -> None:
        logger.warning(f"Lost contact with job for `{table_name}`.")

    def _validate_gretel_model(self, gretel_model: Optional[str]) -> str:
        if gretel_model is None:
            gretel_model = self._strategy.default_model
        gretel_model = gretel_model.lower()

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

        return _BLUEPRINTS[gretel_model]


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


def _cautiously_refresh_status(
    job: Job, key: str, refresh_attempts: Dict[str, int]
) -> Status:
    try:
        job.refresh()
        refresh_attempts[key] = 0
    except:
        refresh_attempts[key] = refresh_attempts[key] + 1

    return job.status
