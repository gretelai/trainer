import os
import pandas as pd
from pathlib import Path
import random
from typing import Any, Dict, List, Optional

from gretel_trainer import Trainer
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
        self.models = {}
        os.makedirs(self.working_dir, exist_ok=True)

    def _prepare_training_data(self, rdb_config: Dict[str, Any]) -> dict:
        # Remove all primary and foreign key fields from the training data
        # Start by gathering the columns for each table
        table_fields_use = {}
        for table in rdb_config["table_data"]:
            table_fields_use[table] = list(rdb_config["table_data"][table].columns)

        # Now, loop through the primary/foreign key relations and gather those columns
        primary_keys_processed = []
        for key_set in rdb_config["relationships"]:
            for table_field_pair in key_set:
                table, field = table_field_pair
                if field==rdb_config["primary_keys"][table]:
                    primary_keys_processed.append(table)
                table_fields_use[table].remove(field)

        # Gather the remaining primary keys
        for table in rdb_config["primary_keys"]:
            if table not in primary_keys_processed:
                field = rdb_config["primary_keys"][table]
                table_fields_use[table].remove(field)

        # Remove the key fields from the training data
        training_data = {}
        for table in rdb_config["table_data"]:
            train_df = rdb_config["table_data"][table].filter(table_fields_use[table])
            training_path = self.working_dir / f"{table}-train.csv"
            train_df.head(MAX_ROWS).to_csv(training_path, index=False)
            training_data[table] = training_path

        return training_data

    # nee fit
    def train(self):
        """Train synthetic data models on each table in the relational dataset"""

        training_data = self._prepare_training_data(self.config)
        for table, training_csv in training_data.items():
            model_cache = self.working_dir / f"{table}-runner.json"
            model_name = str(model_cache)
            self.models[table] = model_name

            print(f"Fitting model: {table}")
            trainer = Trainer(
                model_type=None,
                project_name=f"{self.project_prefix}-{table.replace('_', '-')}",
                cache_file=model_cache,
                overwrite=False,
            )
            trainer.train(training_csv)

    # nee sample
    def generate(self, record_size_ratio: float = 1.0) -> dict:
        """Sample synthetic data from trained models

        Args:
            record_size_ratio (int, optional): Ratio to upsample real world data size with. Defaults to 1.

        Returns:
            dict(pd.DataFrame): Return a dictionary of table names and synthetic data.
        """
        # Compute the number of records needed for each table
        synth_record_counts = {}
        synthetic_tables = {}

        for table in self.config["table_data"]:
            source_df = self.config["table_data"][table]
            if table in self.tables_not_to_synthesize:
                synthetic_tables[table] = source_df
            else:
                train_size = len(source_df)
                synth_size = train_size * record_size_ratio
                synth_record_counts[table] = synth_size

                print(f"Sampling {synth_size} rows from {table}")
                model = Trainer.load(
                    cache_file=self.models[table],
                    project_name=f"{self.project_prefix}-{table.replace('_', '-')}",
                )
                data = model.generate(num_records=synth_record_counts[table])
                synthetic_tables[table] = data

        synthetic_tables = self._synthesize_keys(synthetic_tables, self.config)

        return synthetic_tables

    def _synthesize_keys(
        self,
        synthetic_tables: dict,
        rdb_config: dict,
    ) -> dict:
        # Recompute the number of records needed for each table
        synth_primary_keys = {}
        synth_foreign_keys = {}
        synth_record_counts = {}
        for table_name, df in synthetic_tables.items():
            synth_record_counts[table_name] = len(df)
            synth_foreign_keys[table_name] = {}

        # Synthesize primary keys by assigning a new unique int
        for table_name, field_name in rdb_config["primary_keys"].items():
            df = synthetic_tables[table_name]
            synth_size = synth_record_counts[table_name]
            new_key = [i for i in range(synth_size)]
            synth_primary_keys[table_name] = new_key
            df[field_name] = new_key
            synthetic_tables[table_name] = df

        # Synthesize foreign keys
        for relationship in rdb_config["relationships"]:
            for table_field_pair in relationship:
                rel_table, rel_field = table_field_pair
                # Check if the table/field pair is the primary key
                if rel_field==rdb_config["primary_keys"][rel_table]:
                    primary_key_values = synth_primary_keys[rel_table]
                else:
                    # Now recreate the foreign key values using the primary key values while
                    # preserving the number of records with the same foreign key value
                    # Primary key values range from 0 to size of table holding primary key

                    # Get the frequency distribution of this foreign key
                    table_df = rdb_config["table_data"][rel_table]
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
