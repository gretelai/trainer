import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine.base import Engine
from sqlalchemy.exc import OperationalError

from gretel_trainer.relational.core import MultiTableException, RelationalData

logger = logging.getLogger(__name__)


class Connector:
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

    def extract(
        self, only: Optional[List[str]] = None, ignore: Optional[List[str]] = None
    ) -> RelationalData:
        """
        Extracts table data and relationships from the database.
        To scope to a subset of a database, use either `only` (inclusive) or `ignore` (exclusive).
        """
        if only is not None and ignore is not None:
            raise MultiTableException("Cannot specify both `only` and `ignore`.")

        inspector = inspect(self.engine)

        relational_data = RelationalData()
        foreign_keys: List[Tuple[str, dict]] = []

        for table_name in inspector.get_table_names():
            if _skip_table(table_name, only, ignore):
                continue

            logger.debug(f"Extracting source data from `{table_name}`")
            df = pd.read_sql_table(table_name, self.engine)
            primary_key = inspector.get_pk_constraint(table_name)["constrained_columns"]
            for fk in inspector.get_foreign_keys(table_name):
                if _skip_table(fk["referred_table"], only, ignore):
                    continue
                else:
                    foreign_keys.append((table_name, fk))

            relational_data.add_table(name=table_name, primary_key=primary_key, data=df)

        for foreign_key in foreign_keys:
            table, fk = foreign_key
            relational_data.add_foreign_key_constraint(
                table=table,
                constrained_columns=fk["constrained_columns"],
                referred_table=fk["referred_table"],
                referred_columns=fk["referred_columns"],
            )

        return relational_data

    def save(self, tables: Dict[str, pd.DataFrame], prefix: str = "") -> None:
        for name, data in tables.items():
            data.to_sql(
                f"{prefix}{name}", con=self.engine, if_exists="replace", index=False
            )


def _skip_table(
    table: str, only: Optional[List[str]], ignore: Optional[List[str]]
) -> bool:
    skip = False
    if only is not None and table not in only:
        skip = True
    if ignore is not None and table in ignore:
        skip = True

    return skip


def sqlite_conn(path: str) -> Connector:
    engine = create_engine(f"sqlite:///{path}")
    return Connector(engine)


def postgres_conn(
    *, user: str, password: str, host: str, port: int, database: str
) -> Connector:
    conn_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    engine = create_engine(conn_string)
    return Connector(engine)


def mysql_conn(*, user: str, password: str, host: str, port: int, database: str):
    conn_string = f"mysql://{user}:{password}@{host}:{port}/{database}"
    engine = create_engine(conn_string)
    return Connector(engine)


def mariadb_conn(*, user: str, password: str, host: str, port: int, database: str):
    conn_string = f"mariadb://{user}:{password}@{host}:{port}/{database}"
    engine = create_engine(conn_string)
    return Connector(engine)


def snowflake_conn(
    *,
    user: str,
    password: str,
    account_identifier: str,
    database: str,
    schema: str,
    warehouse: str,
    role: str,
) -> Connector:
    conn_string = f"snowflake://{user}:{password}@{account_identifier}/{database}/{schema}?warehouse={warehouse}&role={role}"
    engine = create_engine(conn_string)
    return Connector(engine)
