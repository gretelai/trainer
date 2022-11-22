import os

import pandas as pd
from sqlalchemy import MetaData
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from collections import defaultdict
from pathlib import Path

from typing import Any, Dict, List, Tuple

from gretel_trainer.relational.core import RelationalData


class _Connection:
    """
    Class to wrap connections to relational databases and backups.
    Connectors should be derived from this class

    Args:
        db_path (str): This URL follows RFC-1738, and usually can include username,
            password, hostname, database name as well as optional keyword arguments
            for additional configuration. In some cases a file path is accepted.
    """

    def __init__(self, db_path: str, out_dir: str):
        self.db_path = db_path
        self.out_dir = Path(out_dir)
        self.engine = create_engine(self.db_path)
        print("Connecting to database")
        try:
            self.engine.connect()
        except OperationalError as e:
            print(f"{e}, {e.__cause__}")
        print("Successfully connected to db")
        self.metadata = MetaData()
        self.metadata.reflect(self.engine)

    def extract(self) -> RelationalData:
        relational_data = RelationalData()
        foreign_keys: List[Tuple[str, str]] = []

        for table_name, table in self.metadata.tables.items():
            df = pd.read_sql_table(table_name, self.engine)
            primary_key = None
            for column in table.columns:
                if column.primary_key:
                    primary_key = column.name
                for f_key in column.foreign_keys:
                    referenced_table = f_key.column.table.name
                    referenced_column = f_key.column.name
                    foreign_keys.append((f"{table_name}.{column.name}", f"{referenced_table}.{referenced_column}"))
            relational_data.add_table(table_name, primary_key, df)

        for foreign_key_tuple in foreign_keys:
            foreign_key, referencing = foreign_key_tuple
            relational_data.add_foreign_key(foreign_key, referencing)

        return relational_data

    def crawl_db(self) -> Dict[str, Any]:
        table_data = {}
        table_files = {}
        primary_keys = {}
        # Dict[Tuple[str, str], List[Tuple[str, str]]]  # where tuple elements are (tablename, columnname)
        rels_by_pkey = defaultdict(list)
        relationships = []  # List[List[Tuple[str, str]]]

        for table_name, table in self.metadata.tables.items():
            df = pd.read_sql_table(table_name, self.engine)
            table_data[table_name] = df
            filepath = self.out_dir / f"{table_name}.csv"
            table_files[table_name] = filepath
            for column in table.columns:
                if column.primary_key:
                    primary_keys[table_name] = column.name
                for f_key in column.foreign_keys:
                    rels_by_pkey[(f_key.column.table.name, f_key.column.name)].append(
                        (table_name, column.name)
                    )

        for p_key, f_keys in rels_by_pkey.items():
            relationships.append([p_key] + f_keys)

        rdb_config = {
            "table_data": table_data,
            "table_files": table_files,
            "primary_keys": primary_keys,
            "relationships": relationships,
        }
        return rdb_config

    def save_to_db(self, synthetic_tables: Dict[str, pd.DataFrame]) -> None:
        pass


class SQLite(_Connection):
    """
    Connector to load data from SQLite databases
    """


class PostgreSQL(_Connection):
    """
    Connector to load data from Postgres databases
    """
