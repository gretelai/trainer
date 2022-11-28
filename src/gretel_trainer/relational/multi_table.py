import os
import random

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from gretel_trainer import Trainer
from gretel_trainer.models import GretelACTGAN
from gretel_trainer.relational.core import RelationalData


class MultiTable:
    """
    Relational data support for the Trainer SDK

    Args:
        relational_data (RelationalData): Core data structure representing the source tables and their relationships.
        tables_not_to_synthesize (list[str], optional): List of tables to skip sampling and leave as they are.
        project_prefix (str, optional): Common prefix for Gretel projects created by this model. Defaults to "multi-table".
        working_dir (str, optional): Directory in which temporary assets should be cached. Defaults to "working".
        max_threads (int, optional): Max number of Trainer jobs (train, generate) to run at once. Defaults to 5.
    """

    def __init__(
        self,
        relational_data: RelationalData,
        tables_not_to_synthesize: Optional[List[str]] = None,
        project_prefix: str = "multi-table",
        working_dir: str = "working",
        max_threads: int = 5,
    ):
        self.project_prefix = project_prefix
        self.relational_data = relational_data
        self.tables_not_to_synthesize = tables_not_to_synthesize or []
        self.working_dir = Path(working_dir)
        self.synthetic_tables = {}
        self.model_cache_files: Dict[str, Path] = {}
        os.makedirs(self.working_dir, exist_ok=True)
        self.thread_pool = ThreadPoolExecutor(max_threads)
        self.futures = []

    def _prepare_training_data(self, tables: List[str]) -> Dict[str, Path]:
        """
        Exports a copy of each table with all primary and foreign key columns removed
        to the working directory. Returns a dict with table names as keys and Paths
        to the CSVs as values.
        """
        training_data = {}
        for table_name in tables:
            columns_to_drop = []
            primary_key = self.relational_data.get_primary_key(table_name)
            if primary_key is not None:
                columns_to_drop.append(primary_key)
            foreign_keys = self.relational_data.get_foreign_keys(table_name)
            columns_to_drop.extend(
                [foreign_key.column_name for foreign_key in foreign_keys]
            )
            training_path = self.working_dir / f"{table_name}-train.csv"
            data = self.relational_data.get_table_data(table_name)
            data.drop(columns=columns_to_drop).to_csv(training_path, index=False)
            training_data[table_name] = training_path

        return training_data

    def _create_trainer_models(self, training_data: Dict[str, Path]) -> None:
        """
        Submits each training CSV in the working directory to Trainer for model creation/training.
        Stores each model's Trainer cache file for Trainer to load later.
        """
        for table_name, training_csv in training_data.items():
            model_cache = self.working_dir / f"{table_name}-runner.json"
            self.model_cache_files[table_name] = model_cache

            print(f"Fitting model: {table_name}")
            trainer = Trainer(
                model_type=GretelACTGAN(),
                project_name=f"{self.project_prefix}-{table_name.replace('_', '-')}",
                cache_file=model_cache,
                overwrite=False,
            )
            self.futures.append(self.thread_pool.submit(trainer.train, training_csv))
        [future.result() for future in self.futures]

    def train(self):
        """Train synthetic data models on each table in the relational dataset"""
        training_data = self._prepare_training_data(self.relational_data.list_all_tables())
        self._create_trainer_models(training_data)

    def retrain_with_table(self, table: str, primary_key: Optional[str], data: pd.DataFrame):
        """
        Provide updated table information and retrain. This method overwrites the table data in the
        `RelationalData` instance. It should be used when initial training fails and source data
        needs to be altered, but progress on other tables can be left as-is.
        """
        # TODO: once we do training with ancestral data, retrain all child tables as well.
        self.relational_data.add_table(table, primary_key, data)
        self.model_cache_files[table].unlink(missing_ok=True)
        training_data = self._prepare_training_data([table])
        self._create_trainer_models(training_data)

    def generate(self, record_size_ratio: float = 1.0) -> Dict[str, pd.DataFrame]:
        """Sample synthetic data from trained models

        Args:
            record_size_ratio (float, optional): Ratio to upsample real world data size with. Defaults to 1.

        Returns:
            dict(pd.DataFrame): Return a dictionary of table names and synthetic data.
        """
        output_tables = {}

        for table_name in self.relational_data.list_all_tables():
            source_data = self.relational_data.get_table_data(table_name)
            if table_name in self.tables_not_to_synthesize:
                output_tables[table_name] = source_data
            else:
                synth_size = int(len(source_data) * record_size_ratio)
                print(f"Sampling {synth_size} rows from {table_name}")
                model = Trainer.load(
                    cache_file=str(self.model_cache_files[table_name]),
                    project_name=f"{self.project_prefix}-{table_name.replace('_', '-')}",
                )
                data = model.generate(num_records=synth_size)
                output_tables[table_name] = data

        return self._synthesize_keys(output_tables)

    def _synthesize_keys(
        self,
        output_tables: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.DataFrame]:
        output_tables = self._synthesize_primary_keys(output_tables)
        output_tables = self._synthesize_foreign_keys(output_tables)
        return output_tables

    def _synthesize_primary_keys(
        self,
        output_tables: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.DataFrame]:
        """
        Alters primary key columns on all tables *except* those flagged by the user as
        not to be synthesized. Assumes the primary key column is of type integer.
        """
        for table_name, out_data in output_tables.items():
            if table_name in self.tables_not_to_synthesize:
                continue

            primary_key = self.relational_data.get_primary_key(table_name)
            if primary_key is None:
                continue

            out_df = output_tables[table_name]
            out_df[primary_key] = [i for i in range(len(out_data))]
            output_tables[table_name] = out_df

        return output_tables

    def _synthesize_foreign_keys(
        self,
        output_tables: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.DataFrame]:
        """
        Alters foreign key columns on all tables (*including* those flagged as not to
        be synthesized to ensure joining to a synthesized parent table continues to work)
        by replacing foreign key column values with valid values from the parent table column
        being referenced.
        """
        for table_name, out_data in output_tables.items():
            for foreign_key in self.relational_data.get_foreign_keys(table_name):
                out_df = output_tables[table_name]

                valid_values = list(
                    output_tables[foreign_key.parent_table_name][
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

        return output_tables


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
