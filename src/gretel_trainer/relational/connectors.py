"""
This module provides the "Connector" class which allows for reading from
and writing to databases and data warehouses. This class can handle
metadata and table extraction automatically. When this is done with
the "Connector.extract" method, a "RelationalData" instance is provided
which you can then use with the "MultiTable" class to process data with
Gretel Transforms, Classify, Synthetics, or a combination of both.
"""
from __future__ import annotations

import logging

from pathlib import Path
from typing import Optional

import pandas as pd

from sqlalchemy import create_engine, inspect
from sqlalchemy.engine.base import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.sql import text

from gretel_trainer.relational.core import (
    DEFAULT_RELATIONAL_SOURCE_DIR,
    MultiTableException,
    RelationalData,
)
from gretel_trainer.relational.extractor import ExtractorConfig, TableExtractor

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

    @classmethod
    def from_conn_str(cls, conn_str: str) -> Connector:
        """
        Alternate constructor that creates a Connector instance
        directly from a connection string.

        Args:
            conn_str: A full connection string for the target database.
        """
        engine = create_engine(conn_str)
        return cls(engine)

    def extract(
        self,
        only: Optional[set[str]] = None,
        ignore: Optional[set[str]] = None,
        schema: Optional[str] = None,
        config: Optional[ExtractorConfig] = None,
        storage_dir: str = DEFAULT_RELATIONAL_SOURCE_DIR,
    ) -> RelationalData:
        """
        Extracts table data and relationships from the database. Optional args include:

        Args:
            only: Only extract these table names, cannot be used with `ignore`
            ignore: Skip extracting these table names, cannot be used with `only`
            schema: An optional schema name that is passed through to SQLAlchemy, may only
                be used with certain dialects.
            config: An optional extraction config. This config can be used to only include
                specific tables, ignore specific tables, and configure subsetting. Please
                see the `ExtractorConfig` docs for more details.
            storage_dir: The output directory where extracted data is stored.
        """
        if only is not None and ignore is not None:
            raise MultiTableException("Cannot specify both `only` and `ignore`.")

        if config is None:
            config = ExtractorConfig(
                only=only, ignore=ignore, schema=schema  # pyright: ignore
            )

        storage_dir_path = Path(storage_dir)
        storage_dir_path.mkdir(parents=True, exist_ok=True)

        extractor = TableExtractor(
            config=config, connector=self, storage_dir=storage_dir_path
        )
        extractor.sample_tables()

        # We ensure to re-create RelationalData after extraction so
        # we can account for any embedded JSON. This also loads
        # each table as a DF in the object which is currently
        # the expected behavior for later operations.
        extractor._relational_data = extractor._create_rel_data(
            extracted_tables=extractor.table_order
        )

        return extractor.relational_data

    def save(
        self,
        output_tables: dict[str, pd.DataFrame],
        prefix: str = "",
        source_relational_data: Optional[RelationalData] = None,
    ) -> None:
        # Default to unsorted if source_relational_data isn't provided
        ordered_source_tables = output_tables.keys()
        if source_relational_data is not None:
            # Depencencies can exist between tables due to foreign key constraints
            # This topologically sorted dependency graph allows for deleting/writing tables in the proper order
            ordered_source_tables = (
                source_relational_data.list_tables_parents_before_children()
            )
        # Tables may not be in output_tables if they failed to train/generate
        table_collection = [
            tbl for tbl in ordered_source_tables if tbl in output_tables
        ]

        logger.info("\n--- Executing SQL insertion for output tables ---")
        logger.info("Emptying output tables if they already exist.")

        # Traverse dependency graph from leaf up to parent
        for source_table_name in reversed(table_collection):
            # Check that this table was not dropped with an "only" param
            # and check if table exists in destination db.
            if inspect(self.engine).has_table(f"{prefix}{source_table_name}"):
                logger.info(
                    f"Table: {prefix}{source_table_name} - Exists in destination db. Deleting all records from table."
                )
                # Not all SQL DBs support table truncation with FK constraints (ie. MSSQL)
                self.engine.execute(
                    text(f"delete from {prefix}{source_table_name}").execution_options(
                        autocommit=True
                    )
                )
            else:
                logger.info(
                    f"Table: {prefix}{source_table_name} - Does not exist in destination db. Skipping record deletion."
                )

        logger.info("\nWriting data to destination.")

        # Traverse dependency graph from parent down to leaf
        for source_table_name in table_collection:
            logger.info(f"Table: {prefix}{source_table_name} - Writing data.")
            data = output_tables[source_table_name]
            data.to_sql(
                f"{prefix}{source_table_name}",
                con=self.engine,
                if_exists="append",
                index=False,
            )


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
