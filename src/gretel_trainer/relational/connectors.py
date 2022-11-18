import os

import pandas as pd
from sqlalchemy import MetaData, text
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from collections import defaultdict
from pathlib import Path

from typing import Any, Dict, Union, Tuple

from gretel_trainer.relational.core import (
    ForeignKey,
    MultiTableException,
    PrimaryKey,
    Table,
    SyntheticTables,
    Source,
)


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
        self.engine = None

    def connect(self) -> None:
        if self.engine is None:
            raise MultiTableException("Missing sqlalchemy engine")
        try:
            self.engine.connect()
            print("Successfully connected to db")
        except OperationalError as e:
            print(f"{e}, {e.__cause__}")

    def crawl_db(self) -> Tuple[Dict[str, Any], Source]:
        metadata = MetaData()
        metadata.reflect(self.engine)

        table_data = {}
        table_files = {}
        primary_keys = {}
        # Dict[Tuple[str, str], List[Tuple[str, str]]]  # where tuple elements are (tablename, columnname)
        rels_by_pkey = defaultdict(list)
        relationships = []  # List[List[Tuple[str, str]]]
        tables = {}
        # relationships_typed = ...

        for table_name, table in metadata.tables.items():
            df = pd.read_sql_table(table_name, self.engine)
            table_data[table_name] = df
            filepath = self.out_dir / f"{table_name}.csv"
            table_files[table_name] = filepath
            primary_key = None
            foreign_keys = []
            for column in table.columns:
                if column.primary_key:
                    primary_key = PrimaryKey(
                        table_name=table_name, column_name=column.name
                    )
                    primary_keys[table_name] = column.name
                for f_key in column.foreign_keys:
                    rels_by_pkey[(f_key.column.table.name, f_key.column.name)].append(
                        (table_name, column.name)
                    )
                    foreign_keys.append(
                        ForeignKey(
                            table_name=table_name,
                            column_name=column.name,
                            references=PrimaryKey(
                                table_name=f_key.column.table.name,
                                column_name=f_key.column.name,
                            ),
                        )
                    )
            tables[table_name] = Table(
                name=table_name,
                data=df,
                path=filepath,
                primary_key=primary_key,
                foreign_keys=foreign_keys,
            )

        for p_key, f_keys in rels_by_pkey.items():
            relationships.append([p_key] + f_keys)

        rdb_config = {
            "table_data": table_data,
            "table_files": table_files,
            "primary_keys": primary_keys,
            "relationships": relationships,
        }
        source = Source(tables=tables)
        return (rdb_config, source)

    # TODO: constrain type to just SyntheticTables post-refactor
    def save_to_db(
        self, synthetic_tables: Union[SyntheticTables, Dict[str, pd.DataFrame]]
    ) -> None:
        pass


class SQLite(_Connection):
    """
    Connector to load data from SQLite databases
    """

    def __init__(self, db_path: str, out_dir: str):
        super().__init__(db_path=db_path, out_dir=out_dir)

        print("Connecting to database")
        self.engine = create_engine(self.db_path)
        self.connect()


class PostgreSQL(_Connection):
    """
    Connector to load data from Postgres databases
    """

    def __init__(self, db_path: str, out_dir: str):
        super().__init__(db_path=db_path, out_dir=out_dir)

        print("Connecting to database")
        self.engine = create_engine(self.db_path)
        self.connect()
