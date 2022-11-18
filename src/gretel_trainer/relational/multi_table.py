import os
import pandas as pd
from pathlib import Path
import random
from typing import Any, Dict, List, Optional

from gretel_trainer import Trainer
from gretel_trainer.models import GretelACTGAN
from gretel_trainer.relational.core import Source


WORKING_DIR = "working"
MAX_ROWS = 1000  # FIXME FOR TEST/DEV ONLY


class MultiTable:
    """
    Relational data support for the Trainer SDK

    Args:
        config (dict): source metadata and tables
        project_name (str, optional): Gretel project name. Defaults to "multi-table".
    """

    def __init__(
        self,
        config: Dict[str, Any],
        source: Source,
        tables_not_to_synthesize: Optional[List[str]] = None,
        project_prefix: str = "multi-table",
        working_dir: str = "working",
    ):
        self.project_prefix = project_prefix
        self.config = config
        self.source = source
        self.tables_not_to_synthesize = tables_not_to_synthesize or []
        self.working_dir = Path(working_dir)
        self.synthetic_tables = {}
        self.model_cache_files: Dict[str, Path] = {}
        os.makedirs(self.working_dir, exist_ok=True)

    def _prepare_training_data(self) -> Dict[str, Path]:
        """
        Exports a copy of each table with all primary and foreign key columns removed
        to the working directory. Returns a dict with table names as keys and Paths
        to the CSVs as values.
        """
        training_data = {}
        for name, table in self.source.tables.items():
            columns_to_drop = []
            if table.primary_key is not None:
                columns_to_drop.append(table.primary_key.column_name)
            columns_to_drop.extend(
                [foreign_key.column_name for foreign_key in table.foreign_keys]
            )
            training_path = self.working_dir / f"{name}-train.csv"
            table.data.drop(columns=columns_to_drop).to_csv(training_path, index=False)
            training_data[name] = training_path

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
            trainer.train(training_csv)

    def train(self):
        """Train synthetic data models on each table in the relational dataset"""
        training_data = self._prepare_training_data()
        self._create_trainer_models(training_data)

    def generate(self, record_size_ratio: float = 1.0) -> Dict[str, pd.DataFrame]:
        """Sample synthetic data from trained models

        Args:
            record_size_ratio (float, optional): Ratio to upsample real world data size with. Defaults to 1.

        Returns:
            dict(pd.DataFrame): Return a dictionary of table names and synthetic data.
        """
        # Compute the number of records needed for each table
        synthetic_tables = {}

        for table_name, table in self.source.tables.items():
            if table_name in self.tables_not_to_synthesize:
                synthetic_tables[table] = table.data
            else:
                synth_size = int(len(table.data) * record_size_ratio)
                print(f"Sampling {synth_size} rows from {table_name}")
                model = Trainer.load(
                    cache_file=str(self.model_cache_files[table_name]),
                    project_name=f"{self.project_prefix}-{table_name.replace('_', '-')}",
                )
                data = model.generate(num_records=synth_size)
                synthetic_tables[table] = data

        self._synthesize_keys(synthetic_tables)

        return synthetic_tables

    def _synthesize_keys_from_source(
        self,
        synthetic_tables: Dict[str, pd.DataFrame],
    ) -> None:
        for table_name, df in synthetic_tables.items():
            if table_name in self.tables_not_to_synthesize:
                continue

            pk = self.source.tables[table_name].primary_key
            if pk is not None:
                df[pk.column_name] = [i for i in range(len(df))]

            for fk in self.source.tables[table_name].foreign_keys:
                df[fk.column_name] = ["TODO"]

    def _synthesize_keys(
        self,
        synthetic_tables: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.DataFrame]:
        # Recompute the number of records needed for each table
        synth_primary_keys = {}
        synth_foreign_keys = {}
        synth_record_counts = {}
        for table_name, df in synthetic_tables.items():
            synth_record_counts[table_name] = len(df)
            synth_foreign_keys[table_name] = {}

        # Synthesize primary keys by assigning a new unique int
        for table_name, field_name in self.config["primary_keys"].items():
            df = synthetic_tables[table_name]
            synth_size = synth_record_counts[table_name]
            new_key = [i for i in range(synth_size)]
            synth_primary_keys[table_name] = new_key
            df[field_name] = new_key
            synthetic_tables[table_name] = df

        # Synthesize foreign keys
        for relationship in self.config["relationships"]:
            for table_field_pair in relationship:
                rel_table, rel_field = table_field_pair
                # Check if the table/field pair is the primary key
                if rel_field == self.config["primary_keys"][rel_table]:
                    primary_key_values = synth_primary_keys[rel_table]
                else:
                    # Now recreate the foreign key values using the primary key values while
                    # preserving the number of records with the same foreign key value
                    # Primary key values range from 0 to size of table holding primary key

                    # Get the frequency distribution of this foreign key
                    table_df = self.config["table_data"][rel_table]
                    freqs = table_df.groupby([rel_field]).size().reset_index()

                    synth_size = synth_record_counts[rel_table]
                    key_values = []
                    key_cnt = 0

                    # Process one primary key at a time. Keep an index on the foreign key freq values
                    # and repeat the primary key as a foreign key for the next freq value.  If we're
                    # increasing the size of the tables, we'll have to loop through the foreign key
                    # values multiple times

                    next_freq_index = 0
                    for i in range(len(primary_key_values)):
                        if next_freq_index == len(freqs):
                            next_freq_index = 0
                        freq = freqs.loc[next_freq_index][0]
                        for j in range(freq):
                            key_values.append(i)
                            key_cnt += 1
                        next_freq_index += 1

                    # Make sure we have reached the desired size of the foreign key table
                    # If not, loop back through the primary keys filling it in.
                    i = 0
                    while key_cnt < synth_size:
                        key_values.append(i)
                        key_cnt += 1
                        i += 1
                    random.shuffle(key_values)
                    synth_foreign_keys[rel_table][rel_field] = key_values

        for table_name, foreign_keys in synth_foreign_keys.items():
            df = synthetic_tables[table_name]
            for key_name, synthetic_keys in foreign_keys.items():
                df[key_name] = synthetic_keys[0 : len(df)]
            synthetic_tables[table_name] = df

        return synthetic_tables
