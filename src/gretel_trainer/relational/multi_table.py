import os
import pandas as pd
import random

from gretel_trainer.relational.connectors import SQLite


WORKING_DIR = "working"


class MultiTable:
    """
    Relational data support for the Trainer SDK

    Args:
        db_path (str): connection string for database path
        project_name (str, optional): Gretel project name. Defaults to "multi-table".
    """

    def __init__(
        self,
        db_path: str,
        project_name: str = "multi-table",
    ):
        print("Initializing connection to database")
        self.db = SQLite(db_path=db_path, working_dir=WORKING_DIR)
        self.db.crawl_db()
        self.synthetic_tables = {}

        if not os.path.exists(WORKING_DIR):
            os.makedirs(WORKING_DIR)

    def _prepare_training_data(self, rdb_config: dict):
        # Remove all primary and foreign key fields from the training data
        # Start by gathering the columns for each table
        table_fields = {}
        table_fields_use = {}
        for table in rdb_config["table_data"]:
            table_fields[table] = list(rdb_config["table_data"][table].columns)
            table_fields_use[table] = list(table_fields[table])

        # Now, loop through the primary/foreign key relations and gather those columns
        primary_keys_processed = []
        for key_set in rdb_config["relationships"]:
            first = True
            for table_field_pair in key_set:
                table, field = table_field_pair
                if first:
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
            table_train = rdb_config["table_data"][table].filter(
                table_fields_use[table]
            )
            training_data[table] = table_train

        return training_data

    def fit(self):
        # TODO: Trainer synthetic code goes here
        # For now, just repeating source data
        self.models = self._prepare_training_data(self.db.config)
        for table, table_df in self.models.items():
            print(f"Fitting model: {table}")

    def sample(self, record_size_ratio=1):
        # Compute the number of records needed for each table
        self.db.config["synth_record_size_ratio"] = record_size_ratio
        self.synth_record_counts = {}
        synthetic_tables = {}

        for table in self.db.config["table_data"]:
            df = self.db.config["table_data"][table]
            train_size = len(df)
            synth_size = train_size * self.db.config["synth_record_size_ratio"]
            self.synth_record_counts[table] = synth_size

            print(f"Sampling {synth_size} rows from {table}")
            data = pd.concat([self.models[table]] * record_size_ratio)
            synthetic_tables[table] = data

        synthetic_tables = self._synthesize_keys(synthetic_tables, self.db.config)

        return synthetic_tables

    def _synthesize_keys(
        self,
        synthetic_tables: dict,
        rdb_config: dict,
    ):
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
            # The first table/field pair is the primary key
            first = True
            for table_field_pair in relationship:
                rel_table, rel_field = table_field_pair
                if first:
                    primary_key_values = synth_primary_keys[rel_table]
                    first = False
                else:
                    # Find the average number of records with the same foreign key value
                    synth_size = synth_record_counts[rel_table]
                    avg_cnt_key = int(synth_size / len(primary_key_values))
                    key_values = []
                    key_cnt = 0
                    # Now recreate the foreign key values using the primary key values while
                    # preserving the avg number of records with the same foreign key value
                    for i in range(len(primary_key_values)):
                        for j in range(avg_cnt_key):
                            key_values.append(i)
                            key_cnt += 1
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
