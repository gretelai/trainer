"""
Extraction of SQL tables to compressed flat files with optional subsetting.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional

import dask.dataframe as dd
import numpy as np
import pandas as pd
from sqlalchemy import MetaData, Table, func, inspect, select, tuple_

from gretel_trainer.relational.core import RelationalData

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from gretel_trainer.relational.connectors import Connector

logger = logging.getLogger(__name__)


class SampleMode(str, Enum):
    RANDOM = "random"
    CONTIGUOUS = "contiguous"


@dataclass
class ExtractorConfig:
    target_row_count: float = -1.0
    """
    The target number of rows (or ratio of rows) to sample. This will be used as the sample target for "leaf" tables, 
    or tables that do not have any references to their primary keys. If this number is >= 1 then
    that number of rows will be used, if the value is between 0..1 then it is considered to be a percetange
    of the total number of rows. A 0 value will just extract headers and -1 will extract entire tables.

    The default value, -1, implies that full tables should be extracted.
    """

    sample_mode: SampleMode = SampleMode.RANDOM

    only: Optional[set[str]] = None
    """
    Only extract these tables. Cannot be used with `ignore.`
    """

    ignore: Optional[set[str]] = None
    """
    Ignore these tables during extraction. Cannot be used with `only.`
    """

    schema: str | None = None
    """
    Limit scope to a specific schema, this is a pass-through param to SQLAlchemy. It is not
    supported by all dialects
    """

    def __post_init__(self):
        errors = []

        if self.sample_mode != SampleMode.RANDOM:
            raise NotImplementedError(
                "Additional sampling modes are not enabled yet, please use `random`"
            )

        if self.target_row_count < -1:
            errors.append("The `target_row_count` must be -1 or higher")

        if self.ignore is not None and self.only is not None:
            errors.append("Cannot specify both `only` and `ignore` together")

        if self.sample_mode not in ("random", "contiguous"):
            errors.append("`sample_mode` must be one of 'random', 'contiguous'")

        if errors:
            raise ValueError(f"The following errors occured: {', '.join(errors)}")

    @property
    def entire_table(self) -> bool:
        return self.target_row_count == -1

    @property
    def empty_table(self) -> bool:
        return self.target_row_count == 0

    def should_skip_table(self, table_name: str) -> bool:
        if self.only and table_name not in self.only:
            return True

        if self.ignore and table_name in self.ignore:
            return True

        return False


def _determine_sample_size(config: ExtractorConfig, total_row_count: int) -> int:
    """
    Given the actual total row count of a table, determine how
    many rows we should sample from it.
    """
    if config.target_row_count >= 1:
        return int(config.target_row_count)

    if config.entire_table:
        return total_row_count

    return int(total_row_count * config.target_row_count)


@dataclass
class _TableSession:
    table: Table
    engine: Engine

    @property
    def total_row_count(self) -> int:
        with self.engine.connect() as conn:
            query = select(func.count()).select_from(self.table)
            count = conn.execute(query).scalar()
            return 0 if count is None else int(count)

    @property
    def total_column_count(self) -> int:
        return len(self.columns)

    @property
    def columns(self) -> list[str]:
        return [column.name for column in self.table.columns]


@dataclass
class _PKValues:
    """
    Contains information that is needed to sample rows from an intermediate table
    """

    table_name: str
    values_ddf: dd.DataFrame  # pyright: ignore
    column_names: list[str]


@dataclass
class TableMetadata:
    """
    Contains information about an extracted table.
    """

    original_row_count: int
    sampled_row_count: int
    column_count: int

    def dict(self) -> dict[str, int]:
        return asdict(self)


def _stream_df_to_path(df_iter: Iterator[pd.DataFrame], path: Path) -> int:
    """
    Stream the contents of a DF to disk, this function only does appending
    """
    row_count = 0

    for df in df_iter:
        df.to_csv(path, mode="a", index=False, header=False)
        row_count += len(df)

    return row_count


class TableExtractorError(Exception):
    pass


class TableExtractor:
    _connector: Connector
    _config: ExtractorConfig
    _storage_dir: Path
    _relational_data: RelationalData
    _chunk_size: int

    table_order: list[str]

    def __init__(
        self,
        *,
        config: ExtractorConfig,
        connector: Connector,
        storage_dir: Path,
    ):
        self._connector = connector
        self._config = config

        if not storage_dir.is_dir():
            raise ValueError("The `storage_dir` must be a directory!")

        self._storage_dir = storage_dir

        self._relational_data = RelationalData()
        self.table_order = []
        self._chunk_size = 50_000

    def _get_table_session(self, table_name: str) -> _TableSession:
        metadata = MetaData()
        metadata.reflect(only=[table_name], bind=self._connector.engine)
        table = metadata.tables[table_name]
        return _TableSession(table=table, engine=self._connector.engine)

    def _create_rel_data(
        self, extracted_tables: Optional[list[str]] = None
    ) -> RelationalData:
        """
        Internal helper method. This can be used to construct a `RelationalData`
        object that either contains just the table headers and FK/PK relationships
        or create an instance that has loaded DataFrames.

        You may need to use this in order to build up a "fresh" `RelationalData` object
        _after_ tables have already been sampled. Especially if you need to consider
        any embedded JSON data that is used to create additional PK/FK mappings
        that are invented.

        If any table names are provided in the `tables` list, then those tables
        will be loaded as DFs and added as the data to nodes.

        NOTE: If `tables` are provided, then these tables must have already been
        extracted!
        """
        if extracted_tables is None:
            extracted_tables = []

        rel_data = RelationalData()
        inspector = inspect(self._connector.engine)
        foreign_keys: list[tuple[str, dict]] = []

        for table_name in inspector.get_table_names(schema=self._config.schema):
            if self._config.should_skip_table(table_name):
                continue

            logger.debug(f"Extracting source schema data from `{table_name}`")

            if table_name not in extracted_tables:
                df = pd.DataFrame(
                    columns=[col["name"] for col in inspector.get_columns(table_name)]
                )
            else:
                df = self.get_table_df(table_name)

            primary_key = inspector.get_pk_constraint(table_name)["constrained_columns"]
            for fk in inspector.get_foreign_keys(table_name):
                if self._config.should_skip_table(fk["referred_table"]):
                    continue
                foreign_keys.append((table_name, fk))

            rel_data.add_table(name=table_name, primary_key=primary_key, data=df)

        for foreign_key in foreign_keys:
            table, fk = foreign_key
            rel_data.add_foreign_key_constraint(
                table=table,
                constrained_columns=fk["constrained_columns"],
                referred_table=fk["referred_table"],
                referred_columns=fk["referred_columns"],
            )

        return rel_data

    def _extract_schema(self) -> None:
        # This will initially only populate RelationalData with empty
        # DataFrames which are only used for building up the right order
        # to extract tables for subsetting purposes. There will be no
        # actual table contents stored on the Graph. This means that
        # after this runs the `RelationalData` object will not have
        # any relationships that may exist from embedded JSON.

        self._relational_data = self._create_rel_data()

        # Set the table processing order for extraction
        self.table_order = list(
            reversed(self._relational_data.list_tables_parents_before_children())
        )

    def _table_path(self, table_name: str) -> Path:
        return self._storage_dir / f"{table_name}.csv"

    def get_table_df(self, table_name: str) -> pd.DataFrame:
        """
        Return a sampled table as a DataFrame. This assumes tables have
        already been sampled and are stored on disk.

        Args:
            table_name: The name of the table to fetch as a DataFrame.
        """
        table_path = self._table_path(table_name)
        if not table_path.is_file():
            raise ValueError(f"The table name: `{table_name}` does not exist.")

        return pd.read_csv(table_path)

    def _load_table_pk_values(
        self, table_name: str, child_table_names: list[str]
    ) -> _PKValues:
        """
        Given a table name, extract all of the values of the primary key of the
        table as they exist in already sampled tables of this table's children.

        In otherwords, iterate all the children of this table and extract the
        values of the foreign keys that reference this table. The values of
        the FKs will represent all the PK values for this table that
        should be extracted as a subset.

        This function assumes the children table already sampled and stored
        based on the required table ordering needed to completed subsetting.1
        """
        values_ddf = None
        parent_column_names: list[str] = []
        pk_set = set(self._relational_data.get_primary_key(table_name))
        logger.debug(
            f"Extacting primary key values for sampling from table '{table_name}'"
        )

        for child_table_name in child_table_names:
            child_fks = self._relational_data.get_foreign_keys(child_table_name)
            for fk in child_fks:
                if fk.parent_table_name == table_name and pk_set == set(
                    fk.parent_columns
                ):
                    if not parent_column_names:
                        parent_column_names = fk.parent_columns
                    logger.debug(
                        f"Found primary key values for table '{table_name}' in child table '{child_table_name}'"
                    )

                    # NOTE: When we extract the FK values from the child tables, we store them under the PK
                    # column names for the current intermediate table we are processing.
                    rename_map = dict(zip(fk.columns, fk.parent_columns))

                    # NOTE: The child tables MUST have alraedy been extracted!
                    child_table_path = self._table_path(child_table_name)

                    tmp_ddf = dd.read_csv(  # pyright: ignore
                        str(child_table_path), usecols=fk.columns
                    )
                    tmp_ddf = tmp_ddf.rename(columns=rename_map)
                    if values_ddf is None:
                        values_ddf = tmp_ddf
                    else:
                        values_ddf = dd.concat([values_ddf, tmp_ddf])  # pyright: ignore

        if parent_column_names:
            values_ddf = values_ddf[  # pyright: ignore
                parent_column_names
            ].drop_duplicates()  # pyright: ignore

        return _PKValues(
            table_name=table_name,
            values_ddf=values_ddf,  # pyright: ignore
            column_names=parent_column_names,
        )

    def _sample_pk_values(self, table_path: Path, pk_values: _PKValues) -> int:
        """
        Given a DDF of PK values for a table, we query for those rows and start
        streaming them to the target path. This assumes the target file already
        exists with column names and we will be appending to that file.
        """
        row_count = 0

        def handle_partition(df: pd.DataFrame):
            # This runs in another thread so we have to re-create our table session info
            table_session = self._get_table_session(pk_values.table_name)
            nonlocal row_count

            chunk_size = 15_000  # limit how many checks go into a WHERE clause

            for _, chunk_df in df.groupby(np.arange(len(df)) // chunk_size):
                values_list = chunk_df.to_records(index=False).tolist()
                query = table_session.table.select().where(
                    tuple_(
                        *[table_session.table.c[col] for col in pk_values.column_names]
                    ).in_(values_list)
                )

                with table_session.engine.connect() as conn:
                    df_iter = pd.read_sql_query(query, conn, chunksize=self._chunk_size)
                    write_count = _stream_df_to_path(df_iter, table_path)
                    row_count += write_count

        logger.debug(
            f"Sampling primary key values for intermediate table '{pk_values.table_name}'"
        )
        pk_values.values_ddf.map_partitions(handle_partition).compute()

        return row_count

    def _flat_sample(
        self, table_path: Path, table_session: _TableSession
    ) -> TableMetadata:
        sample_row_count = _determine_sample_size(
            self._config, table_session.total_row_count
        )

        # TODO: Re-instate when other modes are added
        # if self._config.sample_mode == SampleMode.RANDOM:
        query = (
            select(table_session.table).order_by(func.random()).limit(sample_row_count)
        )

        logger.debug(
            f"Sampling {sample_row_count} rows from table '{table_session.table.name}'"
        )

        with table_session.engine.connect() as conn:
            df_iter = pd.read_sql_query(query, conn, chunksize=self._chunk_size)
            sampled_row_count = _stream_df_to_path(df_iter, table_path)

        return TableMetadata(
            original_row_count=table_session.total_row_count,
            sampled_row_count=sampled_row_count,
            column_count=table_session.total_column_count,
        )

    def _sample_table(
        self, table_name: str, child_tables: Optional[list[str]] = None
    ) -> TableMetadata:
        if self._relational_data.is_empty:
            self._extract_schema()

        table_path = self._table_path(table_name)
        table_session = self._get_table_session(table_name)
        engine = self._connector.engine

        # First we'll create our table file on disk and bootstrap
        # it with just the column names
        df = pd.DataFrame(columns=table_session.columns)
        df.to_csv(table_path, index=False)

        # If we aren't sampling any rows, we're done!
        if self._config.empty_table:
            return TableMetadata(
                original_row_count=table_session.total_row_count,
                sampled_row_count=0,
                column_count=table_session.total_column_count,
            )

        # If we are sampling the entire table, we can just short circuit here
        # and start streaing data into the file
        if self._config.entire_table:
            logger.debug(f"Extracting entire table: {table_name}")
            with engine.connect() as conn:
                df_iter = pd.read_sql_table(
                    table_name, conn, chunksize=self._chunk_size
                )
                sampled_count = _stream_df_to_path(df_iter, table_path)

            return TableMetadata(
                original_row_count=table_session.total_row_count,
                sampled_row_count=sampled_count,
                column_count=table_session.total_column_count,
            )

        # If this is a leaf table, determine how many rows to sample and
        # run the query and start streaming the results
        if not child_tables:
            return self._flat_sample(table_path, table_session)

        # Child nodes exist at this point.

        # If this is an intermediate table, first we build a DDF that contains
        # all of the PK values that we will sample from this intermediate table.
        # These PK values is the set intersection of all the FK values of this
        # intermediate table's child tables
        pk_values = self._load_table_pk_values(table_name, child_tables)
        sampled_row_count = self._sample_pk_values(table_path, pk_values)

        return TableMetadata(
            original_row_count=table_session.total_row_count,
            sampled_row_count=sampled_row_count,
            column_count=table_session.total_column_count,
        )

    def sample_tables(
        self, refresh_relational_data: bool = False
    ) -> dict[str, TableMetadata]:
        """
        Extract database tables according to the `ExtractorConfig.` Tables will be stored in the
        configured storage directory from the `ExtractorConfig` object.

        Args:
            refresh_relational_data: If this is set, the `RelationalData` object will be re-created
                using the extracted tables. This is useful if there is nested JSON in the tables because
                the invented keys for the embedded JSON will now be extracted and accounted for in
                the learned schema.
        """
        if self._relational_data.is_empty:
            self._extract_schema()

        table_data = {}
        for table_name in self.table_order:
            child_tables = self._relational_data.get_descendants(table_name)
            meta = self._sample_table(table_name, child_tables=child_tables)
            table_data[table_name] = meta

        if refresh_relational_data:
            self._relational_data = self._create_rel_data(
                extracted_tables=self.table_order
            )

        return table_data

    @property
    def relational_data(self) -> RelationalData:
        if self._relational_data.is_empty:
            raise TableExtractorError(
                "Cannot return `RelationalData`, `sample_tables()` must be run first."
            )
        return self._relational_data
