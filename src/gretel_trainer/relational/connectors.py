import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import MetaData, create_engine
from sqlalchemy.engine.base import Engine
from sqlalchemy.exc import OperationalError

from gretel_trainer.relational.core import RelationalData

logger = logging.getLogger(__name__)


class Connection:
    """
    Wraps connections to relational databases and backups.

    Args:
        engine (sqlalchemy.engine.base.Engine): A SQLAlchemy engine configured
            to connect to some database. A variety of helper functions exist to
            assist with creating engines for some popular databases, but these
            should not be considered exhaustive. You may need to install
            additional dialect/adapter packages via pip, such as psycopg2 for
            connecting to postgres.

            For more detail, see the SQLAlchemy docs:
            https://docs.sqlalchemy.org/en/20/core/engines.html
    """

    def __init__(self, engine: Engine):
        self.engine = engine
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
            data.to_sql(
                f"{prefix}{name}", con=self.engine, if_exists="replace", index=False
            )


def sqlite_conn(path: str) -> Connection:
    engine = create_engine(f"sqlite:///{path}")
    return Connection(engine)


def postgres_conn(user: str, password: str, host: str, port: int) -> Connection:
    engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}")
    return Connection(engine)


def bigquery_conn(
    project: Optional[str] = None,
) -> Connection:
    engine = create_engine(f"bigquery://{project}")
    return Connection(engine)


def snowflake_conn(
    user: str,
    password: str,
    account_identifier: str,
    database: Optional[str] = None,
    schema: Optional[str] = None,
    warehouse: Optional[str] = None,
    role: Optional[str] = None,
) -> Connection:
    conn_string = f"snowflake:///{user}:{password}@{account_identifier}"

    if database is not None:
        conn_string = f"{conn_string}/{database}"
        if schema is not None:
            conn_string = f"{conn_string}/{schema}"

    next_sep = "?"
    if warehouse is not None:
        conn_string = f"{conn_string}{next_sep}warehouse={warehouse}"
        next_sep = "&"
    if role is not None:
        conn_string = f"{conn_string}{next_sep}role={role}"

    engine = create_engine(conn_string)
    return Connection(engine)
