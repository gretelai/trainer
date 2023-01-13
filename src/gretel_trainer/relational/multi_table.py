import json
import logging
import os
import random
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
from sklearn import preprocessing

from gretel_trainer.relational.core import (
    MultiTableException,
    RelationalData,
    TableEvaluation,
    TblEval,
)
from gretel_trainer.relational.strategies.cross_table import CrossTableStrategy
from gretel_trainer.relational.strategies.single_table import SingleTableStrategy

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
        gretel_model (str, optional): The underlying Gretel model to use. Supports "Amplify" (default), "LSTM", and "ACTGAN".
        correlation_strategy (str, optional): The strategy to use. Supports "cross-table" (default) and "single-table".
        working_dir (str, optional): Directory in which temporary assets should be cached. Defaults to "working".
        refresh_interval (int, optional): Frequency in seconds to poll Gretel Cloud for job statuses. Must be at least 60 (1m). Defaults to 180 (3m).
    """

    def __init__(
        self,
        relational_data: RelationalData,
        project_name: str = "multi-table",
        gretel_model: str = "amplify",
        correlation_strategy: str = "cross-table",
        working_dir: str = "working",
        refresh_interval: Optional[int] = None,
    ):
        gretel_model = gretel_model.lower()
        strategy = correlation_strategy.lower()
        _ensure_valid_combination(gretel_model, strategy)
        self._model_config = _select_model_config(gretel_model)
        self._strategy = _select_strategy(correlation_strategy, gretel_model)

        configure_session(api_key="prompt", cache="yes", validate=True)
        self._project_name = project_name
        self._project = create_or_get_unique_project(name=self._project_name)

        self.relational_data = relational_data
        self._set_refresh_interval(refresh_interval)
        self._models = {}
        self.train_statuses = {}
        self._reset_train_statuses(self.relational_data.list_all_tables())
        self._reset_generation_statuses()
        self._reset_output_tables()
        self.evaluations = defaultdict(lambda: TblEval())

        self._working_dir = Path(working_dir)
        os.makedirs(self._working_dir, exist_ok=True)
        self._create_debug_summary()

    @property
    def state_by_action(self) -> Dict[str, Dict[str, Any]]:
        return {
            "train": self.train_statuses,
            "generate": self.generate_statuses,
            "output": self.output_tables,
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
            "output": self.output_tables.get(table_name),
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
        in_place: bool,
    ) -> Dict[str, pd.DataFrame]:
        """
        Applies supplied transform model configurations to tables. Returned dictionary includes all transformed
        tables, which may not include all known tables (i.e. if a transform config was not provided).

        Args:
            configs (dict[str, GretelModelConfig]): keys are table names and values are Transform model configs.
            in_place (bool): If True, overwrites internal source dataframes with transformed dataframes,
            which means subsequent synthetic model training would be performed on the transform results.

        Returns:
            dict[str, pd.DataFrame]: keys are table names and values are transformed tables
        """
        output_tables = self._execute_transform_jobs(configs)
        output_tables = self._transform_keys(output_tables)

        if in_place:
            for table_name, transformed_table in output_tables:
                self.relational_data.update_table_data(table_name, transformed_table)

        return output_tables

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
            time.sleep(self._refresh_interval)

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

    def _transform_keys(
        self, tables: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Crawls tables for all key columns (primary and foreign) and runs each through a LabelEncoder.
        """
        all_keys = set()
        for table_name in tables:
            primary_key = self.relational_data.get_primary_key(table_name)
            if primary_key is not None:
                all_keys.add((table_name, primary_key))
            for foreign_key in self.relational_data.get_foreign_keys(table_name):
                all_keys.add(
                    (foreign_key.parent_table_name, foreign_key.parent_column_name)
                )

        for key in all_keys:
            table, column = key
            column_data = self.relational_data.get_table_data(table)[column]
            values = list(set(column_data))
            le = preprocessing.LabelEncoder()
            le.fit(values)
            tables[table][column] = le.transform(column_data)

        return tables

    def _prepare_training_data(self, tables: List[str]) -> Dict[str, Path]:
        """
        Exports a copy of each table prepared for training by the configured strategy
        to the working directory. Returns a dict with table names as keys and Paths
        to the CSVs as values.
        """
        training_data = {}
        for table_name in tables:
            training_path = self._working_dir / f"{table_name}_train.csv"
            data = self._strategy.prepare_training_data(
                table_name, self.relational_data
            )
            data.to_csv(training_path, index=False)
            training_data[table_name] = training_path

        return training_data

    def _table_model_config(self, table_name: str) -> Dict:
        config_dict = read_model_config(self._model_config)
        config_dict["name"] = table_name
        return config_dict

    def _train_all(self, training_data: Dict[str, Path]) -> None:
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
            time.sleep(self._refresh_interval)

            for table_name, model in self._models.items():
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

                status = _cautiously_refresh_status(model, table_name, refresh_attempts)

                if status == Status.COMPLETED:
                    self._log_success(table_name, "model training")
                    self.train_statuses[table_name] = TrainStatus.Completed
                    self._strategy.update_evaluation_from_model(
                        self.evaluations[table_name], model
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
        self._train_all(training_data)

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
        self._train_all(training_data)

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
        self._reset_output_tables()

        preserve_tables = preserve_tables or []
        self._skip_some_tables(preserve_tables)

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
                time.sleep(self._refresh_interval)

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
                    self.output_tables[table_name] = _get_data_from_record_handler(
                        record_handler
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
                    self.output_tables,
                )
                self._log_start(table_name, "synthetic data generation")
                self.generate_statuses[table_name] = GenerateStatus.InProgress
                model = self._models[table_name]
                record_handler = model.create_record_handler_obj(**table_job)
                record_handler.submit_cloud()
                record_handlers[table_name] = record_handler

        self._synthesize_keys(preserve_tables)
        return self.output_tables

    def export_csvs(self, prefix: str = "out_") -> None:
        """
        Exports output tables as CSVs to the working directory.
        """
        for table_name, df in self.output_tables.items():
            df.to_csv(f"{self._working_dir}/{prefix}{table_name}.csv", index=False)

    def expand_evaluations(self) -> None:
        """
        Adds evaluation metrics for the "opposite" correlation strategy using the Gretel Evaluate API.
        """
        for table_name in self.output_tables:
            logger.info(
                f"Expanding evaluation metrics for `{table_name}` via Gretel Evaluate API."
            )
            self._strategy.update_evaluation_via_evaluate(
                evaluation=self.evaluations[table_name],
                table=table_name,
                rel_data=self.relational_data,
                synthetic_tables=self.output_tables,
            )

    def evaluate(
        self, synthetic_tables: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Dict[str, TableEvaluation]:
        synthetic_tables = synthetic_tables or self.output_tables
        evaluations = {}

        for table_name, synthetic_data in synthetic_tables.items():
            logger.info(
                f"Evaluating individual SQS and cross-table SQS for `{table_name}`."
            )

            model_sqs_score = self._get_model_sqs_score(table_name)
            evaluation = self._strategy.evaluate(
                table_name, self.relational_data, model_sqs_score, synthetic_tables
            )
            evaluations[table_name] = evaluation

            logger.info(
                f"SQS evaluation for `{table_name}` complete. Individual: {evaluation.individual_sqs}. Cross-table: {evaluation.cross_table_sqs}."
            )

        return evaluations

    def _get_model_sqs_score(self, table_name: str) -> Optional[int]:
        model = self._models.get(table_name)
        if model is None:
            return None

        summary = model.get_report_summary()
        if summary is None or summary.get("summary") is None:
            return None

        sqs_score = None
        for stat in summary["summary"]:
            if stat["field"] == "synthetic_data_quality_score":
                sqs_score = stat["value"]

        return sqs_score

    def _reset_generation_statuses(self) -> None:
        """
        Sets the GenerateStatus for all known tables to NotStarted.
        """
        self.generate_statuses = {
            table_name: GenerateStatus.NotStarted
            for table_name in self.relational_data.list_all_tables()
        }

    def _reset_output_tables(self) -> None:
        "Clears all output tables"
        self.output_tables = {}

    def _skip_some_tables(self, preserve_tables: List[str]) -> None:
        "Updates state for tables being preserved and tables lacking trained models."
        for table in self.relational_data.list_all_tables():
            if table in preserve_tables:
                self.generate_statuses[table] = GenerateStatus.SourcePreserved
                self.output_tables[table] = self.relational_data.get_table_data(table)
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

    def _synthesize_keys(self, preserve_tables: List[str]) -> Dict[str, pd.DataFrame]:
        self.output_tables = self._synthesize_primary_keys(preserve_tables)
        self.output_tables = self._synthesize_foreign_keys()
        return self.output_tables

    def _synthesize_primary_keys(
        self, preserve_tables: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """
        Alters primary key columns on all tables *except* those flagged by the user as
        not to be synthesized. Assumes the primary key column is of type integer.
        """
        for table_name, out_data in self.output_tables.items():
            if table_name in preserve_tables:
                continue

            primary_key = self.relational_data.get_primary_key(table_name)
            if primary_key is None:
                continue

            out_df = self.output_tables[table_name]
            out_df[primary_key] = [i for i in range(len(out_data))]
            self.output_tables[table_name] = out_df

        return self.output_tables

    def _synthesize_foreign_keys(self) -> Dict[str, pd.DataFrame]:
        """
        Alters foreign key columns on all tables (*including* those flagged as not to
        be synthesized to ensure joining to a synthesized parent table continues to work)
        by replacing foreign key column values with valid values from the parent table column
        being referenced.
        """
        for table_name, out_data in self.output_tables.items():
            for foreign_key in self.relational_data.get_foreign_keys(table_name):
                out_df = self.output_tables[table_name]

                valid_values = list(
                    self.output_tables[foreign_key.parent_table_name][
                        foreign_key.parent_column_name
                    ]
                )

                original_table_data = self.relational_data.get_table_data(table_name)
                original_fk_frequencies = (
                    original_table_data.groupby(foreign_key.column_name)
                    .size()
                    .reset_index()
                )
                frequencies_descending = sorted(
                    list(original_fk_frequencies[0]), reverse=True
                )

                new_fk_values = _collect_new_foreign_key_values(
                    valid_values, frequencies_descending, len(out_df)
                )

                out_df[foreign_key.column_name] = new_fk_values

        return self.output_tables

    def _table_generation_in_progress(self, table: str) -> bool:
        return self.generate_statuses[table] == GenerateStatus.InProgress

    def _table_generation_in_terminal_state(self, table: str) -> bool:
        return self.generate_statuses[table] in [
            GenerateStatus.Completed,
            GenerateStatus.SourcePreserved,
            GenerateStatus.ModelUnavailable,
            GenerateStatus.Failed,
        ]

    def _log_start(self, table_name: str, action: str) -> None:
        logger.info(
            f"Starting {action} for `{table_name}`. Next status check in {self._refresh_interval} seconds."
        )

    def _log_in_progress(self, table_name: str, action: str) -> None:
        logger.info(
            f"{action.capitalize()} job for `{table_name}` still in progress. Next status check in {self._refresh_interval} seconds."
        )

    def _log_failed(self, table_name: str, action: str) -> None:
        logger.info(f"{action.capitalize()} failed for `{table_name}`.")

    def _log_success(self, table_name: str, action: str) -> None:
        logger.info(f"{action.capitalize()} successfully completed for `{table_name}`.")

    def _log_lost_contact(self, table_name: str) -> None:
        logger.warning(f"Lost contact with job for `{table_name}`.")


def _collect_new_foreign_key_values(
    values: List[Any],
    frequencies: List[int],
    total: int,
) -> List[Any]:
    """
    Uses `random.choices` to select `k=total` elements from `values`,
    weighted according to `frequencies`.
    """
    v_len = len(values)
    f_len = len(frequencies)
    f_iter = iter(frequencies)

    if v_len == f_len:
        # Unlikely but convenient exact match
        weights = frequencies

    elif v_len < f_len:
        # Add as many frequencies as necessary, in descending order
        weights = []
        while len(weights) < v_len:
            weights.append(next(f_iter))

    else:
        # Add all frequencies to start, then fill in more as necessary in descending order
        weights = frequencies
        while len(weights) < v_len:
            weights.append(next(f_iter))

    random.shuffle(weights)
    return random.choices(values, weights=weights, k=total)


def _get_data_from_record_handler(record_handler: RecordHandler) -> pd.DataFrame:
    return pd.read_csv(record_handler.get_artifact_link("data"), compression="gzip")


def _ensure_valid_combination(model: str, strategy: str) -> None:
    if strategy == "cross-table":
        if model != "amplify":
            msg = f"Cross-table strategy does not support {model}; only amplify is supported."
            logger.warning(msg)
            raise MultiTableException(msg)


def _select_model_config(model: str) -> str:
    if model == "amplify":
        return "synthetics/amplify"
    elif model == "lstm":
        return "synthetics/tabular-lstm"
    elif model == "actgan":
        return "synthetics/tabular-actgan"
    else:
        msg = f"Unrecognized gretel model requested: {model}. Supported models are `amplify`, `lsmt`, and `actgan`."
        logger.warning(msg)
        raise MultiTableException(msg)


def _select_strategy(
    strategy: str, model: str
) -> Union[SingleTableStrategy, CrossTableStrategy]:
    if strategy == "cross-table":
        return CrossTableStrategy(model)
    elif strategy == "single-table":
        return SingleTableStrategy()
    else:
        msg = f"Unrecognized correlation strategy requested: {strategy}. Supported strategies are `cross-table` and `single-table`."
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
