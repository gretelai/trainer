import json
import logging
import os
import random
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed, wait
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from gretel_client import configure_session
from gretel_client.evaluation.quality_report import QualityReport
from gretel_client.helpers import poll
from gretel_client.projects import create_or_get_unique_project
from sklearn import preprocessing

from gretel_trainer import Trainer
from gretel_trainer.models import GretelAmplify, GretelLSTM
from gretel_trainer.relational.core import MultiTableException, RelationalData, TableEvaluation
from gretel_trainer.relational.strategies.ancestral import AncestralStrategy
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
        gretel_model (str, optional): The underlying Gretel model to use. Supports "Amplify" (default) and "LSTM".
        correlation_strategy (str, optional): The strategy to use. Supports "cross-table" (default) and "single-table".
        working_dir (str, optional): Directory in which temporary assets should be cached. Defaults to "working".
        max_threads (int, optional): Max number of Trainer jobs (train, generate) to run at once. Defaults to 5.
    """

    def __init__(
        self,
        relational_data: RelationalData,
        project_name: str = "multi-table",
        gretel_model: str = "Amplify",
        correlation_strategy: str = "cross-table",
        working_dir: str = "working",
        max_threads: int = 5,
    ):
        configure_session(api_key="prompt", cache="yes", validate=True)
        self._project_name = project_name
        self._project = create_or_get_unique_project(name=self._project_name)
        self._gretel_model = _select_gretel_model(gretel_model)
        self._strategy = _select_strategy(correlation_strategy)
        self.relational_data = relational_data
        self.working_dir = Path(working_dir)
        os.makedirs(self.working_dir, exist_ok=True)
        self._create_debug_summary()
        self.thread_pool = ThreadPoolExecutor(max_threads)
        self.train_statuses = {}
        self._reset_train_statuses(self.relational_data.list_all_tables())
        self._reset_generation_statuses()
        self._reset_output_tables()

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

    def _create_debug_summary(self) -> None:
        debug_summary_path = self.working_dir / "_gretel_debug_summary.json"
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
        output_tables = {}
        transforms_futures = []

        for table_name, config in configs.items():
            logger.info(f"Queueing transforms job for `{table_name}`")
            table_data = self.relational_data.get_table_data(table_name)
            transforms_futures.append(
                self.thread_pool.submit(
                    _transform,
                    table_name,
                    self._project,
                    table_data,
                    config,
                )
            )

        for future in as_completed(transforms_futures):
            table_name, out_table = future.result()
            logger.info(f"Transforms job for `{table_name}` completed")
            output_tables[table_name] = out_table

        output_tables = self._transform_keys(output_tables)

        if in_place:
            for table_name, transformed_table in output_tables:
                self.relational_data.update_table_data(table_name, transformed_table)

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
            training_path = self.working_dir / f"{table_name}.csv"
            data = self._strategy.prepare_training_data(
                table_name, self.relational_data
            )
            data.to_csv(training_path, index=False)
            training_data[table_name] = training_path

        return training_data

    def _create_trainer_models(self, training_data: Dict[str, Path]) -> None:
        """
        Submits each training CSV in the working directory to Trainer for model creation/training.
        Stores each model's Trainer cache file for Trainer to load later.
        """
        train_futures = []
        for table_name, training_csv in training_data.items():
            logger.info(f"Starting model training for `{table_name}`")
            trainer = Trainer(
                model_type=self._gretel_model,
                project_name=self._project_name,
                cache_file=self._cache_file_for(table_name),
                overwrite=False,
            )
            self.train_statuses[table_name] = TrainStatus.InProgress
            train_futures.append(
                self.thread_pool.submit(_train, trainer, training_csv, table_name)
            )

        for future in as_completed(train_futures):
            table_name, successful = future.result()
            if successful:
                logger.info(f"Training successfully completed for `{table_name}`")
                self.train_statuses[table_name] = TrainStatus.Completed
            else:
                logger.info(f"Training failed for `{table_name}`")
                self.train_statuses[table_name] = TrainStatus.Failed

    def train(self) -> None:
        """Train synthetic data models on each table in the relational dataset"""
        tables = self.relational_data.list_all_tables()
        self._reset_train_statuses(tables)

        training_data = self._prepare_training_data(tables)
        self._create_trainer_models(training_data)

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
        self._create_trainer_models(training_data)

    def _reset_train_statuses(self, tables: List[str]) -> None:
        for table in tables:
            self.train_statuses[table] = TrainStatus.NotStarted
            self._cache_file_for(table).unlink(missing_ok=True)

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

        futures: Dict[str, List[Future]] = {}
        while self._more_to_do():
            self._update_generate_statuses(futures)

            in_progress_tables = [
                table
                for table in self.relational_data.list_all_tables()
                if self._table_in_progress(table)
            ]
            finished_tables = [
                table
                for table in self.relational_data.list_all_tables()
                if self._table_in_terminal_state(table)
            ]
            ready_tables = self._strategy.ready_to_generate(
                self.relational_data, in_progress_tables, finished_tables
            )
            for table_name in ready_tables:
                self._start_generation_jobs(table_name, futures, record_size_ratio)

            time.sleep(10)

        # The while loop above ends when there are no jobs remaining with NotStarted status,
        # but some jobs could still be InProgress, so now we wait for everything to finish.
        all_futures_flattened = [
            future for list_of_futures in futures.values() for future in list_of_futures
        ]
        wait(all_futures_flattened)

        # One last call to update state now that all jobs are complete.
        self._update_generate_statuses(futures)

        self._synthesize_keys(preserve_tables)

        return self.output_tables

    def evaluate(
        self, synthetic_tables: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Dict[str, TableEvaluation]:
        synthetic_tables = synthetic_tables or self.output_tables
        evaluations = {}
        evaluate_futures = []
        for table_name, synthetic_data in synthetic_tables.items():
            evaluate_futures.append(
                self.thread_pool.submit(
                    self._evaluate_table,
                    table_name,
                    synthetic_tables,
                )
            )
        for future in as_completed(evaluate_futures):
            table_name, evaluation = future.result()
            evaluations[table_name] = evaluation

        return evaluations

    def _evaluate_table(
        self, table_name: str, synthetic_tables: Dict[str, pd.DataFrame]
    ) -> Tuple[str, TableEvaluation]:
        return table_name, TableEvaluation(
            individual_sqs=self._get_individual_sqs_score(table_name, synthetic_tables),
            ancestral_sqs=self._get_ancestral_sqs_score(table_name, synthetic_tables),
        )

    def _get_individual_sqs_score(
        self, table_name: str, synthetic_tables: Dict[str, pd.DataFrame]
    ) -> int:
        if self.train_statuses[table_name] == TrainStatus.Completed:
            model = self._load_trainer_model(table_name)
            return model.get_sqs_score()
        else:
            return _get_sqs_via_evaluate(
                synthetic_tables[table_name],
                self.relational_data.get_table_data(table_name),
            )

    def _get_ancestral_sqs_score(
        self, table_name: str, synthetic_tables: Dict[str, pd.DataFrame]
    ) -> int:
        ancestral_synthetic_data = self.relational_data.get_table_data_with_ancestors(
            table_name, synthetic_tables
        )
        ancestral_reference_data = self.relational_data.get_table_data_with_ancestors(
            table_name
        )
        return _get_sqs_via_evaluate(ancestral_synthetic_data, ancestral_reference_data)

    def _load_trainer_model(self, table_name: str) -> Trainer:
        return Trainer.load(
            cache_file=str(self._cache_file_for(table_name)),
            project_name=self._project_name,
        )

    def _cache_file_for(self, table_name: str) -> Path:
        return self.working_dir / f"{table_name}-runner.json"

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

    def _start_generation_jobs(
        self,
        table_name: str,
        futures: Dict[str, List[Future]],
        record_size_ratio: float,
    ) -> None:
        logger.info(f"Starting synthetic data generation for `{table_name}`")
        model = self._load_trainer_model(table_name)
        table_jobs = self._strategy.get_generation_jobs(
            table_name,
            self.relational_data,
            record_size_ratio,
            self.output_tables,
        )
        table_job_futures = []
        for job_kwargs in table_jobs:
            table_job_futures.append(
                self.thread_pool.submit(_generate, model, job_kwargs)
            )
        self.generate_statuses[table_name] = GenerateStatus.InProgress
        futures[table_name] = table_job_futures

    def _update_generate_statuses(
        self,
        futures: Dict[str, List[Future]],
    ) -> None:
        for table_name in self.relational_data.list_all_tables():
            if not self._table_in_progress(table_name):
                continue

            table_job_futures = futures[table_name]

            if all([future.done() for future in table_job_futures]):
                self.generate_statuses[table_name] = GenerateStatus.Completed
                component_dataframes = []
                for future in as_completed(futures[table_name]):
                    dataframe = future.result()
                    if dataframe is not None:
                        component_dataframes.append(dataframe)
                if len(component_dataframes) > 0:
                    logger.info(
                        f"Synthetic data generation completed for `{table_name}`"
                    )
                    self.output_tables[
                        table_name
                    ] = self._strategy.collect_generation_results(
                        component_dataframes, table_name, self.relational_data
                    )
                else:
                    logger.info(f"Synthetic data generation failed for `{table_name}`")
                    self.generate_statuses[table_name] = GenerateStatus.Failed

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

    def _table_in_progress(self, table: str) -> bool:
        return self.generate_statuses[table] == GenerateStatus.InProgress

    def _table_in_terminal_state(self, table: str) -> bool:
        return self.generate_statuses[table] in [
            GenerateStatus.Completed,
            GenerateStatus.SourcePreserved,
            GenerateStatus.ModelUnavailable,
            GenerateStatus.Failed,
        ]

    def _more_to_do(self) -> bool:
        return any(
            [
                status == GenerateStatus.NotStarted
                for status in self.generate_statuses.values()
            ]
        )


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


def _transform(
    table_name: str,
    project,
    table_data: pd.DataFrame,
    config: GretelModelConfig,
) -> Tuple[str, pd.DataFrame]:
    model = project.create_model_obj(model_config=config, data_source=table_data)
    model.submit_cloud()

    poll(model)

    record_handler = model.create_record_handler_obj(data_source=table_data)
    record_handler.submit_cloud()

    poll(record_handler)

    return (
        table_name,
        pd.read_csv(record_handler.get_artifact_link("data"), compression="gzip"),
    )


def _train(
    trainer,
    training_csv: Path,
    table_name: str,
) -> Tuple[str, bool]:
    trainer.train(training_csv)
    return (table_name, trainer.trained_successfully())


def _generate(
    model,
    job_kwargs: Dict[str, Any],
) -> pd.DataFrame:
    try:
        return model.generate(**job_kwargs)
    except:
        return None


def _get_sqs_via_evaluate(data_source: pd.DataFrame, ref_data: pd.DataFrame) -> int:
    report = QualityReport(data_source=data_source, ref_data=ref_data)
    report.run()
    return report.peek()["score"]


def _select_gretel_model(model: str) -> Union[GretelAmplify, GretelLSTM]:
    model = model.lower()
    if model == "amplify":
        return GretelAmplify()
    elif model == "lstm":
        return GretelLSTM()
    else:
        raise MultiTableException(
            f"Unrecognized gretel model requested: {model}. Supported models are `Amplify` and `LSTM`."
        )


def _select_strategy(strategy: str) -> Union[SingleTableStrategy, AncestralStrategy]:
    strategy = strategy.lower()
    if strategy == "cross-table":
        return AncestralStrategy()
    elif strategy == "single-table":
        return SingleTableStrategy()
    else:
        raise MultiTableException(
            f"Unrecognized correlation strategy requested: {strategy}. Supported strategies are `cross-table` and `single-table`."
        )
