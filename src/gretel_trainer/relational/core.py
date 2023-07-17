"""
This module exposes the "RelationalData" class to users, which allows the processing
of relational databases and data warehouses with Gretel.ai.

When using a "Connector" or a "TableExtractor" instance to automatically connect
to a database, a "RelationalData" instance will be created for you that contains
all of the learned metadata.

If you are processing relational tables manually, with your own CSVs, you
will need to create a "RelationalData" instance and populate it yourself.

Please see the specific docs for the "RelationalData" class on how to do this.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import networkx
import pandas as pd
from networkx.algorithms.dag import dag_longest_path_length, topological_sort
from pandas.api.types import is_string_dtype

import gretel_trainer.relational.json as relational_json
from gretel_trainer.relational.json import (
    IngestResponseT,
    InventedTableMetadata,
    ProducerMetadata,
)

logger = logging.getLogger(__name__)

DEFAULT_RELATIONAL_SOURCE_DIR = "relational_source"
PREVIEW_ROW_COUNT = 5


class MultiTableException(Exception):
    pass


GretelModelConfig = Union[str, Path, dict]


@dataclass
class ForeignKey:
    table_name: str
    columns: list[str]
    parent_table_name: str
    parent_columns: list[str]


UserFriendlyDataT = Union[pd.DataFrame, str, Path]
UserFriendlyPrimaryKeyT = Optional[Union[str, list[str]]]


class Scope(str, Enum):
    """
    Various non-mutually-exclusive sets of tables known to the system
    """

    ALL = "all"
    """
    Every known table (all user-supplied tables, all invented tables)
    """

    PUBLIC = "public"
    """
    Includes all user-supplied tables, omits invented tables
    """

    MODELABLE = "modelable"
    """
    Includes flat source tables and all invented tables, omits source tables that led to invented tables
    """

    EVALUATABLE = "evaluatable"
    """
    A subset of MODELABLE that additionally omits invented child tables (but includes invented root tables)
    """

    INVENTED = "invented"
    """
    Includes all tables invented from un-modelable user source tables
    """


@dataclass
class TableMetadata:
    primary_key: list[str]
    source: Path
    columns: list[str]
    invented_table_metadata: Optional[InventedTableMetadata] = None
    producer_metadata: Optional[ProducerMetadata] = None
    safe_ancestral_seed_columns: Optional[set[str]] = None


@dataclass
class _RemovedTableMetadata:
    source: Path
    primary_key: list[str]
    fks_to_parents: list[ForeignKey]
    fks_from_children: list[ForeignKey]


class RelationalData:
    """
    Stores information about multiple tables and their relationships. When
    using this object you could create it without any arguments and rely
    on the instance methods for adding tables and key relationships.

    Example::

        rel_data = RelationalData()
        rel_data.add_table(...)
        rel_data.add_table(...)
        rel_data.add_foreign_key_constraint(...)

    See the specific method docstrings for details on each method.
    """

    def __init__(self, directory: Optional[Union[str, Path]] = None):
        self.dir = Path(directory or DEFAULT_RELATIONAL_SOURCE_DIR)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.graph = networkx.DiGraph()

    @property
    def is_empty(self) -> bool:
        """
        Return a bool to indicate if the `RelationalData` contains
        any table information.
        """
        return not self.graph.number_of_nodes() > 0

    def restore(self, tableset: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Restores a given tableset (presumably output from some MultiTable workflow,
        i.e. transforms or synthetics) to its original shape (specifically, "re-nests"
        any JSON that had been expanded out.

        Users should rely on MultiTable calling this internally when appropriate and not
        need to do so themselves.
        """
        restored = {}
        discarded = set()

        # Restore any invented tables to nested-JSON format
        producers = {
            table: pmeta
            for table in self.list_all_tables(Scope.ALL)
            if (pmeta := self.get_producer_metadata(table)) is not None
        }
        for table_name, producer_metadata in producers.items():
            tables = {
                table: data
                for table, data in tableset.items()
                if table in producer_metadata.table_names
            }
            data = relational_json.restore(
                tables=tables,
                rel_data=self,
                root_table_name=producer_metadata.invented_root_table_name,
                original_columns=self.get_table_columns(table_name),
                table_name_mappings=producer_metadata.table_name_mappings,
                original_table_name=table_name,
            )
            if data is not None:
                restored[table_name] = data
            discarded.update(producer_metadata.table_names)

        # Add remaining tables
        for table, data in tableset.items():
            if table not in discarded:
                restored[table] = data

        return restored

    def add_table(
        self,
        *,
        name: str,
        primary_key: UserFriendlyPrimaryKeyT,
        data: UserFriendlyDataT,
    ) -> None:
        """
        Add a table. The primary key can be None (if one is not defined on the table),
        a string column name (most common), or a list of multiple string column names (composite key).

        This call MAY result in multiple tables getting "registered," specifically if
        the table includes nested JSON data.
        """
        primary_key = self._format_key_column(primary_key)
        source_path = Path(f"{self.dir}/{name}.csv")

        # Preview data to get list of columns and determine if there is any JSON
        if isinstance(data, pd.DataFrame):
            preview_df = data.head(PREVIEW_ROW_COUNT)
        elif isinstance(data, (str, Path)):
            preview_df = pd.read_csv(data, nrows=PREVIEW_ROW_COUNT)
        columns = list(preview_df.columns)
        json_cols = relational_json.get_json_columns(preview_df)

        # Write/copy source to preferred internal location
        if isinstance(data, pd.DataFrame):
            relational_json.jsonencode(data, json_cols).to_csv(source_path, index=False)
        elif isinstance(data, (str, Path)):
            shutil.copyfile(data, source_path)

        # If we found JSON in preview above, run JSON ingestion on full data
        rj_ingest = None
        if len(json_cols) > 0:
            logger.info(
                f"Detected JSON data in table `{name}`. Running JSON normalization."
            )
            if isinstance(data, pd.DataFrame):
                df = data
            elif isinstance(data, (str, Path)):
                df = pd.read_csv(data)
            rj_ingest = relational_json.ingest(name, primary_key, df, json_cols)

        # Add the table(s)
        if rj_ingest is not None:
            self._add_producer_and_invented_tables(
                name, primary_key, source_path, columns, rj_ingest
            )
        else:
            self._add_single_table(
                name=name,
                primary_key=primary_key,
                source=source_path,
                columns=columns,
            )

    def _add_producer_and_invented_tables(
        self,
        table: str,
        primary_key: list[str],
        source: Path,
        columns: list[str],
        rj_ingest: IngestResponseT,
    ) -> None:
        commands, producer_metadata = rj_ingest
        tables, foreign_keys = commands

        # Add the producer table
        self._add_single_table(
            name=table,
            primary_key=primary_key,
            source=source,
            columns=columns,
            producer_metadata=producer_metadata,
        )

        # Add the invented tables
        for tbl in tables:
            name = tbl["name"]
            source_path = Path(f"{self.dir}/{name}.csv")
            df = tbl["data"]
            df.to_csv(source_path, index=False)
            self._add_single_table(
                name=name,
                primary_key=tbl["primary_key"],
                source=source_path,
                columns=list(df.columns),
                invented_table_metadata=tbl["invented_table_metadata"],
            )
        for foreign_key in foreign_keys:
            self.add_foreign_key_constraint(**foreign_key)

    def _add_single_table(
        self,
        *,
        name: str,
        primary_key: UserFriendlyPrimaryKeyT,
        source: Path,
        columns: Optional[list[str]] = None,
        invented_table_metadata: Optional[InventedTableMetadata] = None,
        producer_metadata: Optional[ProducerMetadata] = None,
    ) -> None:
        primary_key = self._format_key_column(primary_key)
        columns = columns or list(pd.read_csv(source, nrows=1).columns)
        metadata = TableMetadata(
            primary_key=primary_key,
            source=source,
            columns=columns,
            invented_table_metadata=invented_table_metadata,
            producer_metadata=producer_metadata,
        )
        self.graph.add_node(name, metadata=metadata)

    def _get_table_metadata(self, table: str) -> TableMetadata:
        try:
            return self.graph.nodes[table]["metadata"]
        except KeyError:
            raise MultiTableException(f"Unrecognized table: `{table}`")

    def set_primary_key(
        self, *, table: str, primary_key: UserFriendlyPrimaryKeyT
    ) -> None:
        """
        (Re)set the primary key on an existing table.
        If the table does not yet exist in the instance's collection, add it via `add_table`.
        """
        if table not in self.list_all_tables(Scope.ALL):
            raise MultiTableException(f"Unrecognized table name: `{table}`")

        primary_key = self._format_key_column(primary_key)

        known_columns = self.get_table_columns(table)
        for col in primary_key:
            if col not in known_columns:
                raise MultiTableException(f"Unrecognized column name: `{col}`")

        # Prevent interfering with manually invented tables
        if self._is_invented(table):
            raise MultiTableException("Cannot change primary key on invented tables")

        # If `table` is a producer of invented tables, we redo JSON ingestion
        # to ensure primary keys are set properly on invented tables
        elif self.is_producer_of_invented_tables(table):
            source = self.get_table_source(table)
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpfile = Path(tmpdir) / f"{table}.csv"
                shutil.move(source, tmpfile)
                removal_metadata = self._remove_producer(table)
                self.add_table(name=table, primary_key=primary_key, data=tmpfile)
            self._restore_fks_in_both_directions(table, removal_metadata)

        # At this point we are working with a "normal" table
        else:
            self._get_table_metadata(table).primary_key = primary_key
            self._clear_safe_ancestral_seed_columns(table)

    def _restore_fks_in_both_directions(
        self, table: str, removal_metadata: _RemovedTableMetadata
    ) -> None:
        for fk in removal_metadata.fks_to_parents:
            self.add_foreign_key_constraint(
                table=table,
                constrained_columns=fk.columns,
                referred_table=fk.parent_table_name,
                referred_columns=fk.parent_columns,
            )

        for fk in removal_metadata.fks_from_children:
            self.add_foreign_key_constraint(
                table=fk.table_name,
                constrained_columns=fk.columns,
                referred_table=table,
                referred_columns=fk.parent_columns,
            )

    def _get_user_defined_fks_to_table(self, table: str) -> list[ForeignKey]:
        return [
            fk
            for child in self.graph.predecessors(table)
            for fk in self.get_foreign_keys(child)
            if fk.parent_table_name == table and not self._is_invented(fk.table_name)
        ]

    def _remove_producer(self, table: str) -> _RemovedTableMetadata:
        """
        Removes the producer table and all its invented tables from the graph
        (which in turn removes all edges (foreign keys) to/from other tables).

        Returns a _RemovedTableMetadata object for restoring metadata in broader "update" contexts.
        """
        table_metadata = self._get_table_metadata(table)
        producer_metadata = table_metadata.producer_metadata

        if producer_metadata is None:
            raise MultiTableException(
                "Cannot remove invented tables from non-producer table"
            )

        removal_metadata = _RemovedTableMetadata(
            source=table_metadata.source,
            primary_key=table_metadata.primary_key,
            fks_to_parents=self.get_foreign_keys(table),
            fks_from_children=self._get_user_defined_fks_to_table(
                self._get_fk_delegate_table(table)
            ),
        )

        for invented_table_name in producer_metadata.table_names:
            if invented_table_name in self.graph.nodes:
                self._remove_node(invented_table_name)
        self._remove_node(table)

        return removal_metadata

    def _remove_node(self, table: str) -> None:
        self.graph.remove_node(table)

    def _format_key_column(self, key: Optional[Union[str, list[str]]]) -> list[str]:
        if key is None:
            return []
        elif isinstance(key, str):
            return [key]
        else:
            return key

    def add_foreign_key(self, *, foreign_key: str, referencing: str) -> None:
        """
        DEPRECATED: Please use `add_foreign_key_constraint` instead.

        Format of both str arguments should be `table_name.column_name`
        """
        logger.warning(
            "This method is deprecated and will be removed in a future release. "
            "Please use `add_foreign_key_constraint` instead."
        )
        fk_table, fk_column = foreign_key.split(".")
        referred_table, referred_column = referencing.split(".")
        self.add_foreign_key_constraint(
            table=fk_table,
            constrained_columns=[fk_column],
            referred_table=referred_table,
            referred_columns=[referred_column],
        )

    def add_foreign_key_constraint(
        self,
        *,
        table: str,
        constrained_columns: list[str],
        referred_table: str,
        referred_columns: list[str],
    ) -> None:
        """
        Add a foreign key relationship between two tables.

        Args:
            table: The table name that contains the foreign key.
            constrained_columns: The column name(s) defining a relationship to the `referred_table` (the parent table).
            referred_table: The table name that the foreign key in `table` refers to (the parent table).
            referred_columns: The column name(s) in the parent table that the `constrained_columns` point to.
        """
        known_tables = self.list_all_tables(Scope.ALL)

        abort = False
        if table not in known_tables:
            logger.warning(f"Unrecognized table name: `{table}`")
            abort = True
        if referred_table not in known_tables:
            logger.warning(f"Unrecognized table name: `{referred_table}`")
            abort = True

        if abort:
            raise MultiTableException("Unrecognized table(s) in foreign key arguments")

        if len(constrained_columns) != len(referred_columns):
            logger.warning(
                "Constrained and referred columns must be of the same length"
            )
            raise MultiTableException(
                "Invalid column constraints in foreign key arguments"
            )

        table_all_columns = self.get_table_columns(table)
        for col in constrained_columns:
            if col not in table_all_columns:
                logger.warning(
                    f"Constrained column `{col}` does not exist on table `{table}`"
                )
                abort = True
        referred_table_all_columns = self.get_table_columns(referred_table)
        for col in referred_columns:
            if col not in referred_table_all_columns:
                logger.warning(
                    f"Referred column `{col}` does not exist on table `{referred_table}`"
                )
                abort = True

        if abort:
            raise MultiTableException("Unrecognized column(s) in foreign key arguments")

        fk_delegate_table = self._get_fk_delegate_table(table)
        fk_delegate_referred_table = self._get_fk_delegate_table(referred_table)

        self.graph.add_edge(fk_delegate_table, fk_delegate_referred_table)
        edge = self.graph.edges[fk_delegate_table, fk_delegate_referred_table]
        via = edge.get("via", [])
        via.append(
            ForeignKey(
                table_name=fk_delegate_table,
                columns=constrained_columns,
                parent_table_name=fk_delegate_referred_table,
                parent_columns=referred_columns,
            )
        )
        edge["via"] = via
        self._clear_safe_ancestral_seed_columns(fk_delegate_table)
        self._clear_safe_ancestral_seed_columns(table)

    def remove_foreign_key(self, foreign_key: str) -> None:
        """
        DEPRECATED: Please use `remove_foreign_key_constraint` instead.
        """
        logger.warning(
            "This method is deprecated and will be removed in a future release. "
            "Please use `remove_foreign_key_constraint` instead."
        )
        fk_table, fk_column = foreign_key.split(".")
        self.remove_foreign_key_constraint(
            table=fk_table, constrained_columns=[fk_column]
        )

    def remove_foreign_key_constraint(
        self, table: str, constrained_columns: list[str]
    ) -> None:
        """
        Remove an existing foreign key.
        """
        if table not in self.list_all_tables(Scope.ALL):
            raise MultiTableException(f"Unrecognized table name: `{table}`")

        key_to_remove = None
        for fk in self.get_foreign_keys(table):
            if fk.columns == constrained_columns:
                key_to_remove = fk

        if key_to_remove is None:
            raise MultiTableException(
                f"`{table} does not have a foreign key with constrained columns {constrained_columns}`"
            )

        fk_delegate_table = self._get_fk_delegate_table(table)

        edge = self.graph.edges[fk_delegate_table, key_to_remove.parent_table_name]
        via = edge.get("via")
        via.remove(key_to_remove)
        if len(via) == 0:
            self.graph.remove_edge(fk_delegate_table, key_to_remove.parent_table_name)
        else:
            edge["via"] = via
        self._clear_safe_ancestral_seed_columns(fk_delegate_table)
        self._clear_safe_ancestral_seed_columns(table)

    def update_table_data(self, table: str, data: UserFriendlyDataT) -> None:
        """
        Set a DataFrame as the table data for a given table name.
        """
        if self._is_invented(table):
            raise MultiTableException("Cannot modify invented tables' data")
        elif self.is_producer_of_invented_tables(table):
            removal_metadata = self._remove_producer(table)
        else:
            removal_metadata = _RemovedTableMetadata(
                source=Path(),  # we don't care about the old data
                primary_key=self.get_primary_key(table),
                fks_to_parents=self.get_foreign_keys(table),
                fks_from_children=self._get_user_defined_fks_to_table(table),
            )
            self._remove_node(table)

        self.add_table(name=table, primary_key=removal_metadata.primary_key, data=data)
        self._restore_fks_in_both_directions(table, removal_metadata)

    def list_all_tables(self, scope: Scope = Scope.MODELABLE) -> list[str]:
        """
        Returns a list of table names belonging to the provided Scope.
        See "Scope" enum documentation for details.
        By default, returns tables that can be submitted as jobs to Gretel
        (i.e. that are MODELABLE).
        """
        graph_nodes = list(self.graph.nodes)

        producer_tables = [
            t for t in graph_nodes if self.is_producer_of_invented_tables(t)
        ]

        modelable_tables = []
        evaluatable_tables = []
        invented_tables: list[str] = []

        for n in graph_nodes:
            meta = self._get_table_metadata(n)
            if (invented_meta := meta.invented_table_metadata) is not None:
                invented_tables.append(n)
                if invented_meta.invented_root_table_name == n:
                    evaluatable_tables.append(n)
                if not invented_meta.empty:
                    modelable_tables.append(n)
            else:
                if n not in producer_tables:
                    modelable_tables.append(n)
                    evaluatable_tables.append(n)

        if scope == Scope.MODELABLE:
            return modelable_tables
        elif scope == Scope.EVALUATABLE:
            return evaluatable_tables
        elif scope == Scope.INVENTED:
            return invented_tables
        elif scope == Scope.ALL:
            return graph_nodes
        elif scope == Scope.PUBLIC:
            return [t for t in graph_nodes if t not in invented_tables]

    def _is_invented(self, table: str) -> bool:
        return self.get_invented_table_metadata(table) is not None

    def is_producer_of_invented_tables(self, table: str) -> bool:
        return self.get_producer_metadata(table) is not None

    def get_modelable_table_names(self, table: str) -> list[str]:
        """
        Returns a list of MODELABLE table names connected to the provided table.
        If the provided table is the source of invented tables, returns the modelable invented tables created from it.
        If the provided table is itself modelable, returns that table name back.
        Otherwise returns an empty list.
        """
        try:
            table_metadata = self._get_table_metadata(table)
        except MultiTableException:
            return []

        if (pmeta := table_metadata.producer_metadata) is not None:
            return [
                t
                for t in pmeta.table_names
                if t in self.list_all_tables(Scope.MODELABLE)
            ]
        elif table in self.list_all_tables(Scope.MODELABLE):
            return [table]
        else:
            return []

    def get_public_name(self, table: str) -> Optional[str]:
        if (imeta := self.get_invented_table_metadata(table)) is not None:
            return imeta.original_table_name

        return table

    def get_invented_table_metadata(
        self, table: str
    ) -> Optional[InventedTableMetadata]:
        return self._get_table_metadata(table).invented_table_metadata

    def get_producer_metadata(self, table: str) -> Optional[ProducerMetadata]:
        return self._get_table_metadata(table).producer_metadata

    def get_parents(self, table: str) -> list[str]:
        """
        Given a table name, return the table names that are referred to
        by the foreign keys in this table.
        """
        return list(self.graph.successors(table))

    def get_ancestors(self, table: str) -> list[str]:
        """
        Same as `get_parents` except recursively keep adding
        parent tables until there are no more.
        """

        def _add_parents(ancestors, table):
            parents = self.get_parents(table)
            if len(parents) > 0:
                ancestors.update(parents)
                for parent in parents:
                    _add_parents(ancestors, parent)

        ancestors = set()
        _add_parents(ancestors, table)

        return list(ancestors)

    def get_descendants(self, table: str) -> list[str]:
        """
        Given a table name, recursively return all tables that
        carry foreign keys that reference the primary key in this table
        and all subsequent tables that are discovered.
        """

        def _add_children(descendants, table):
            children = list(self.graph.predecessors(table))
            if len(children) > 0:
                descendants.update(children)
                for child in children:
                    _add_children(descendants, child)

        descendants = set()
        _add_children(descendants, table)

        return list(descendants)

    def list_tables_parents_before_children(self) -> list[str]:
        """
        Returns a list of all tables with the guarantee that a parent table
        appears before any of its children. No other guarantees about order
        are made, e.g. the following (and others) are all valid outputs:
        [p1, p2, c1, c2] or [p2, c2, p1, c1] or [p2, p1, c1, c2] etc.
        """
        return list(reversed(list(topological_sort(self.graph))))

    def get_primary_key(self, table: str) -> list[str]:
        """
        Return the list of columns defining the primary key for a table.
        It may be a single column or multiple columns (composite key).
        """
        return self._get_table_metadata(table).primary_key

    def get_table_source(self, table: str) -> Path:
        return self._get_table_metadata(table).source

    def get_table_data(
        self, table: str, usecols: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """
        Return the table contents for a given table name as a DataFrame.
        """
        source = self.get_table_source(table)
        usecols = usecols or self.get_table_columns(table)
        return pd.read_csv(source, usecols=usecols)

    def get_table_columns(self, table: str) -> list[str]:
        """
        Return the column names for a provided table name.
        """
        return self._get_table_metadata(table).columns

    def get_safe_ancestral_seed_columns(self, table: str) -> set[str]:
        safe_columns = self._get_table_metadata(table).safe_ancestral_seed_columns
        if safe_columns is None:
            safe_columns = self._set_safe_ancestral_seed_columns(table)
        return safe_columns

    def _set_safe_ancestral_seed_columns(self, table: str) -> set[str]:
        cols = set()

        # Key columns are always kept
        cols.update(self.get_primary_key(table))
        for fk in self.get_foreign_keys(table):
            cols.update(fk.columns)

        data = self.get_table_data(table)
        for col in self.get_table_columns(table):
            if col in cols:
                continue
            if _ok_for_train_and_seed(col, data):
                cols.add(col)

        self._get_table_metadata(table).safe_ancestral_seed_columns = cols
        return cols

    def _clear_safe_ancestral_seed_columns(self, table: str) -> None:
        self._get_table_metadata(table).safe_ancestral_seed_columns = None

    def _get_fk_delegate_table(self, table: str) -> str:
        if (pmeta := self.get_producer_metadata(table)) is not None:
            return pmeta.invented_root_table_name

        return table

    def get_foreign_keys(
        self, table: str, rename_invented_tables: bool = False
    ) -> list[ForeignKey]:
        def _rename_invented(fk: ForeignKey) -> ForeignKey:
            table_name = self.get_public_name(fk.table_name)
            parent_table_name = self.get_public_name(fk.parent_table_name)
            return replace(
                fk, table_name=table_name, parent_table_name=parent_table_name
            )

        table = self._get_fk_delegate_table(table)
        foreign_keys = []
        for parent in self.get_parents(table):
            fks = self.graph.edges[table, parent]["via"]
            foreign_keys.extend(fks)

        if rename_invented_tables:
            return [_rename_invented(fk) for fk in foreign_keys]
        else:
            return foreign_keys

    def get_all_key_columns(self, table: str) -> list[str]:
        all_key_cols = []
        all_key_cols.extend(self.get_primary_key(table))
        for fk in self.get_foreign_keys(table):
            all_key_cols.extend(fk.columns)
        return all_key_cols

    def debug_summary(self) -> dict[str, Any]:
        max_depth = dag_longest_path_length(self.graph)
        public_table_count = len(self.list_all_tables(Scope.PUBLIC))
        invented_table_count = len(self.list_all_tables(Scope.INVENTED))

        all_tables = self.list_all_tables(Scope.ALL)
        total_foreign_key_count = 0
        tables = {}
        for table in all_tables:
            this_table_foreign_key_count = 0
            foreign_keys = []
            for key in self.get_foreign_keys(table):
                total_foreign_key_count = total_foreign_key_count + 1
                this_table_foreign_key_count = this_table_foreign_key_count + 1
                foreign_keys.append(
                    {
                        "columns": key.columns,
                        "parent_table_name": key.parent_table_name,
                        "parent_columns": key.parent_columns,
                    }
                )
            table_metadata = {
                "column_count": len(self.get_table_columns(table)),
                "primary_key": self.get_primary_key(table),
                "foreign_key_count": this_table_foreign_key_count,
                "foreign_keys": foreign_keys,
                "is_invented_table": self._is_invented(table),
            }
            if (producer_metadata := self.get_producer_metadata(table)) is not None:
                table_metadata["invented_table_details"] = {
                    "table_type": "producer",
                    "json_to_table_mappings": producer_metadata.table_name_mappings,
                }
            elif (
                invented_table_metadata := self.get_invented_table_metadata(table)
            ) is not None:
                table_metadata["invented_table_details"] = {
                    "table_type": "invented",
                    "json_breadcrumb_path": invented_table_metadata.json_breadcrumb_path,
                }
            tables[table] = table_metadata

        return {
            "foreign_key_count": total_foreign_key_count,
            "max_depth": max_depth,
            "tables": tables,
            "public_table_count": public_table_count,
            "invented_table_count": invented_table_count,
        }


def skip_table(
    table: str, only: Optional[set[str]], ignore: Optional[set[str]]
) -> bool:
    skip = False
    if only is not None and table not in only:
        skip = True
    if ignore is not None and table in ignore:
        skip = True

    return skip


def _ok_for_train_and_seed(col: str, df: pd.DataFrame) -> bool:
    if _is_highly_nan(col, df):
        return False

    if _is_highly_unique_categorical(col, df):
        return False

    return True


def _is_highly_nan(col: str, df: pd.DataFrame) -> bool:
    total = len(df)
    if total == 0:
        return False

    missing = df[col].isnull().sum()
    missing_perc = missing / total
    return missing_perc > 0.2


def _is_highly_unique_categorical(col: str, df: pd.DataFrame) -> bool:
    return is_string_dtype(df[col]) and _percent_unique(col, df) >= 0.7


def _percent_unique(col: str, df: pd.DataFrame) -> float:
    col_no_nan = df[col].dropna()
    total = len(col_no_nan)
    distinct = col_no_nan.nunique()

    if total == 0:
        return 0.0
    else:
        return distinct / total
