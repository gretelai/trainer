import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from sqlalchemy import MetaData, create_engine
from sqlalchemy.exc import OperationalError

from gretel_trainer.relational.core import RelationalData

logger = logging.getLogger(__name__)


class _Connection:
    """
    Class to wrap connections to relational databases and backups.
    Connectors should be derived from this class

    Args:
        db_path (str): This URL follows RFC-1738, and usually can include username,
            password, hostname, database name as well as optional keyword arguments
            for additional configuration. In some cases a file path is accepted.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.engine = create_engine(self.db_path)
        logger.info("Connecting to database")
        try:
            self.engine.connect()
        except OperationalError as e:
            logger.error(f"{e}, {e.__cause__}")
            raise e
        logger.info("Successfully connected to db")

    def extract(self) -> RelationalData:
        """Extracts table data and relationships from the database."""
        metadata = MetaData()
        metadata.reflect(self.engine)

        relational_data = RelationalData()
        foreign_keys: List[Tuple[str, str]] = []

        for table_name, table in metadata.tables.items():
            df = pd.read_sql_table(table_name, self.engine)
            primary_key = None
            for column in table.columns:
                if column.primary_key:
                    primary_key = column.name
                for f_key in column.foreign_keys:
                    referenced_table = f_key.column.table.name
                    referenced_column = f_key.column.name
                    foreign_keys.append(
                        (
                            f"{table_name}.{column.name}",
                            f"{referenced_table}.{referenced_column}",
                        )
                    )
            relational_data.add_table(table_name, primary_key, df)

        for foreign_key_tuple in foreign_keys:
            foreign_key, referencing = foreign_key_tuple
            relational_data.add_foreign_key(foreign_key, referencing)

        return relational_data

    def save(self, tables: Dict[str, pd.DataFrame], prefix: str = "") -> None:
        for name, data in tables.items():
            data.to_sql(f"{prefix}{name}", con=self.engine, if_exists="replace", index=False)


class SQLite(_Connection):
    """
    Connector to/from SQLite databases
    """


class PostgreSQL(_Connection):
    """
    Connector to/from Postgres databases
    """
