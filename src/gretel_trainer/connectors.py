import os

import pandas as pd
from sqlalchemy import MetaData, text
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from collections import defaultdict


class _Connection:
    """
    Class to wrap connections to relational databases and backups.
    Connectors should be derived from this class

    Args:
        db_path (str): This URL follows RFC-1738, and usually can include username,
            password, hostname, database name as well as optional keyword arguments
            for additional configuration. In some cases a file path is accepted.
    """

    def __init__(self, db_path: int, working_dir: str):
        self.db_path = db_path
        self.working_dir = working_dir
        self.engine = None
        self.config = None

    def connect(self):
        try:
            self.engine.connect()
            print("Successfully connected to db")
        except OperationalError as e:
            print(f"{e}, {e.__cause__}")

    def crawl_db(self):

        base_path = "./"

        metadata = MetaData()
        metadata.reflect(self.engine)

        rdb_config = {}
        rdb_config["table_data"] = {}
        rdb_config["table_files"] = {}

        for name, table in metadata.tables.items():
            df = pd.read_sql_table(name, self.engine)
            rdb_config["table_data"][name] = df
            filename = name + ".csv"
            df.to_csv(filename, index=False, header=True)
            rdb_config["table_files"][name] = base_path + filename

        # Extract primary/foriegn key relationshihps
        rels_by_pkey = defaultdict(list)

        for name, table in metadata.tables.items():
            for col in table.columns:
                for f_key in col.foreign_keys:
                    rels_by_pkey[(f_key.column.table.name, f_key.column.name)].append(
                        (name, col.name)
                    )

        list_of_rels_by_pkey = []

        for p_key, f_keys in rels_by_pkey.items():
            list_of_rels_by_pkey.append([p_key] + f_keys)

        rdb_config["relationships"] = list_of_rels_by_pkey

        # Extract each table's primary key
        rdb_config["primary_keys"] = {}

        for name, table in metadata.tables.items():
            for col in table.columns:
                if col.primary_key:
                    rdb_config["primary_keys"][name] = col.name

        self.config = rdb_config

    def save_to_rdb(
        self, orig_db: str, new_db: str, transformed_tables: dict, engine, type="sqlite"
    ):
        print("Type " + type + " is not supported yet")


class SQLite(_Connection):
    """
    Connector to load data from SQLite databases
    """

    def __init__(self, db_path: str, working_dir: str):
        super().__init__(db_path=db_path, working_dir=working_dir)

        print("Connecting to database")
        self.engine = create_engine(self.db_path)
        self.connect()
