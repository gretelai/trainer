from __future__ import annotations

import logging
from contextlib import suppress
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import networkx
import pandas as pd
from networkx.algorithms.dag import dag_longest_path_length, topological_sort
from pandas.api.types import is_string_dtype

from gretel_trainer.relational.json import (
    IngestResponseT,
    InventedTableMetadata,
    RelationalJson,
)

logger = logging.getLogger(__name__)


class MultiTableException(Exception):
    pass


GretelModelConfig = Union[str, Path, Dict]


@dataclass
class ForeignKey:
    table_name: str
    columns: List[str]
    parent_table_name: str
    parent_columns: List[str]


UserFriendlyPrimaryKeyT = Optional[Union[str, List[str]]]


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
    data: pd.DataFrame
    columns: set[str]
    invented_table_metadata: Optional[InventedTableMetadata] = None
    safe_ancestral_seed_columns: Optional[set[str]] = None


class RelationalData:
    def __init__(self):
        self.graph = networkx.DiGraph()
        self.relational_jsons: dict[str, RelationalJson] = {}

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
        for table_name, rel_json in self.relational_jsons.items():
            tables = {
                table: data
                for table, data in tableset.items()
                if table in rel_json.table_names
            }
            data = rel_json.restore(tables, self)
            if data is not None:
                restored[table_name] = data
            discarded.update(rel_json.table_names)

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
        data: pd.DataFrame,
    ) -> None:
        """
        Add a table. The primary key can be None (if one is not defined on the table),
        a string column name (most common), or a list of multiple string column names (composite key).

        This call MAY result in multiple tables getting "registered," specifically if
        the table includes nested JSON data.
        """
        primary_key = self._format_key_column(primary_key)
        rj_ingest = RelationalJson.ingest(name, primary_key, data)
        if rj_ingest is not None:
            self._add_rel_json_and_tables(name, rj_ingest)
        else:
            self._add_single_table(name=name, primary_key=primary_key, data=data)

    def _add_rel_json_and_tables(self, table: str, rj_ingest: IngestResponseT) -> None:
        rel_json, commands = rj_ingest
        tables, foreign_keys = commands

        self.relational_jsons[table] = rel_json

        for tbl in tables:
            self._add_single_table(**tbl)
        for foreign_key in foreign_keys:
            self.add_foreign_key_constraint(**foreign_key)

    def _add_single_table(
        self,
        *,
        name: str,
        primary_key: UserFriendlyPrimaryKeyT,
        data: pd.DataFrame,
        invented_table_metadata: Optional[InventedTableMetadata] = None,
    ) -> None:
        primary_key = self._format_key_column(primary_key)
        metadata = TableMetadata(
            primary_key=primary_key,
            data=data,
            columns=set(data.columns),
            invented_table_metadata=invented_table_metadata,
        )
        self.graph.add_node(name, metadata=metadata)

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

        if self.relational_jsons.get(table) is not None:
            original_data, _, original_fks = self._remove_relational_json(table)
            if original_data is None:
                raise MultiTableException("Original data with JSON is lost.")

            new_rj_ingest = RelationalJson.ingest(table, primary_key, original_data)
            if new_rj_ingest is None:
                raise MultiTableException(
                    "Failed to change primary key on tables invented from JSON data"
                )

            self._add_rel_json_and_tables(table, new_rj_ingest)
            for fk in original_fks:
                self.add_foreign_key_constraint(
                    table=fk.table_name,
                    constrained_columns=fk.columns,
                    referred_table=fk.parent_table_name,
                    referred_columns=fk.parent_columns,
                )
        else:
            self.graph.nodes[table]["metadata"].primary_key = primary_key
            self._clear_safe_ancestral_seed_columns(table)

    def _get_user_defined_fks_to_table(self, table: str) -> list[ForeignKey]:
        return [
            fk
            for child in self.graph.predecessors(table)
            for fk in self.get_foreign_keys(child)
            if fk.parent_table_name == table and not self._is_invented(fk.table_name)
        ]

    def _remove_relational_json(
        self, table: str
    ) -> tuple[Optional[pd.DataFrame], list[str], list[ForeignKey]]:
        """
        Removes the RelationalJson instance and removes all invented tables from the graph.

        Returns the original data, primary key, and foreign keys.
        """
        rel_json = self.relational_jsons[table]

        original_data = rel_json.original_data
        original_primary_key = rel_json.original_primary_key
        original_foreign_keys = self._get_user_defined_fks_to_table(
            rel_json.root_table_name
        )

        for invented_table_name in rel_json.table_names:
            with suppress(KeyError):
                self.graph.remove_node(invented_table_name)
        del self.relational_jsons[table]

        return original_data, original_primary_key, original_foreign_keys

    def _format_key_column(self, key: Optional[Union[str, List[str]]]) -> List[str]:
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
        constrained_columns: List[str],
        referred_table: str,
        referred_columns: List[str],
    ) -> None:
        """
        Add a foreign key relationship between two tables.
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

        table = self._get_table_in_graph(table)
        referred_table = self._get_table_in_graph(referred_table)

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

        self.graph.add_edge(table, referred_table)
        edge = self.graph.edges[table, referred_table]
        via = edge.get("via", [])
        via.append(
            ForeignKey(
                table_name=table,
                columns=constrained_columns,
                parent_table_name=referred_table,
                parent_columns=referred_columns,
            )
        )
        edge["via"] = via
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
        self, table: str, constrained_columns: List[str]
    ) -> None:
        """
        Remove an existing foreign key.
        """
        if table not in self.list_all_tables(Scope.ALL):
            raise MultiTableException(f"Unrecognized table name: `{table}`")

        table = self._get_table_in_graph(table)

        key_to_remove = None
        for fk in self.get_foreign_keys(table):
            if fk.columns == constrained_columns:
                key_to_remove = fk

        if key_to_remove is None:
            raise MultiTableException(
                f"`{table} does not have a foreign key with constrained columns {constrained_columns}`"
            )

        edge = self.graph.edges[table, key_to_remove.parent_table_name]
        via = edge.get("via")
        via.remove(key_to_remove)
        if len(via) == 0:
            self.graph.remove_edge(table, key_to_remove.parent_table_name)
        else:
            edge["via"] = via
        self._clear_safe_ancestral_seed_columns(table)

    def update_table_data(self, table: str, data: pd.DataFrame) -> None:
        if table in self.relational_jsons:
            _, original_pk, original_fks = self._remove_relational_json(table)
            new_rj_ingest = RelationalJson.ingest(table, original_pk, data)
            if new_rj_ingest is not None:
                self._add_rel_json_and_tables(table, new_rj_ingest)
                parent_table_name = new_rj_ingest[0].root_table_name
            else:
                self._add_single_table(
                    name=table,
                    primary_key=original_pk,
                    data=data,
                )
                parent_table_name = table
            for fk in original_fks:
                self.add_foreign_key_constraint(
                    table=fk.table_name,
                    constrained_columns=fk.columns,
                    referred_table=parent_table_name,
                    referred_columns=fk.parent_columns,
                )
        else:
            try:
                metadata = self.graph.nodes[table]["metadata"]
            except KeyError:
                raise MultiTableException(
                    f"Unrecognized table name: {table}. If this is a new table to add, use `add_table`."
                )

            if (
                new_rj_ingest := RelationalJson.ingest(
                    table, metadata.primary_key, data
                )
            ) is not None:
                original_foreign_keys = self._get_user_defined_fks_to_table(table)
                self.graph.remove_node(table)
                self._add_rel_json_and_tables(table, new_rj_ingest)
                for fk in original_foreign_keys:
                    self.add_foreign_key_constraint(
                        table=fk.table_name,
                        constrained_columns=fk.columns,
                        referred_table=new_rj_ingest[0].root_table_name,
                        referred_columns=fk.parent_columns,
                    )
            else:
                metadata.data = data
                metadata.columns = set(data.columns)
                self._clear_safe_ancestral_seed_columns(table)

    def list_all_tables(self, scope: Scope = Scope.MODELABLE) -> List[str]:
        modelable_nodes = self.graph.nodes

        json_source_tables = [
            rel_json.original_table_name
            for _, rel_json in self.relational_jsons.items()
        ]

        if scope == Scope.MODELABLE:
            return list(modelable_nodes)
        elif scope == Scope.EVALUATABLE:
            e = []
            for n in modelable_nodes:
                meta = self.graph.nodes[n]["metadata"]
                if (
                    meta.invented_table_metadata is None
                    or meta.invented_table_metadata.invented_root_table_name == n
                ):
                    e.append(n)
            return e
        elif scope == Scope.INVENTED:
            return [n for n in modelable_nodes if self._is_invented(n)]
        elif scope == Scope.ALL:
            return list(modelable_nodes) + json_source_tables
        elif scope == Scope.PUBLIC:
            non_invented_nodes = [
                n for n in modelable_nodes if not self._is_invented(n)
            ]
            return json_source_tables + non_invented_nodes

    def _is_invented(self, table: str) -> bool:
        return (
            table in self.graph.nodes
            and self.graph.nodes[table]["metadata"].invented_table_metadata is not None
        )

    def get_modelable_table_names(self, table: str) -> list[str]:
        """Returns a list of MODELABLE table names connected to the provided table.
        If the provided table is already modelable, returns [table].
        If the provided table is not modelable (e.g. source with JSON), returns tables invented from that source.
        If the provided table does not exist, returns empty list.
        """
        if (rel_json := self.relational_jsons.get(table)) is not None:
            return rel_json.table_names
        elif table not in self.list_all_tables(Scope.ALL):
            logger.warning(f"Unrecognized table name: `{table}`")
            return []
        else:
            return [table]

    def get_public_name(self, table: str) -> Optional[str]:
        if table in self.relational_jsons:
            return table

        if (
            imeta := self.graph.nodes[table]["metadata"].invented_table_metadata
        ) is not None:
            return imeta.original_table_name

        return table

    def get_invented_table_metadata(
        self, table: str
    ) -> Optional[InventedTableMetadata]:
        if table in self.relational_jsons:
            return None

        return self.graph.nodes[table]["metadata"].invented_table_metadata

    def get_parents(self, table: str) -> List[str]:
        return list(self.graph.successors(table))

    def get_ancestors(self, table: str) -> List[str]:
        def _add_parents(ancestors, table):
            parents = self.get_parents(table)
            if len(parents) > 0:
                ancestors.update(parents)
                for parent in parents:
                    _add_parents(ancestors, parent)

        ancestors = set()
        _add_parents(ancestors, table)

        return list(ancestors)

    def get_descendants(self, table: str) -> List[str]:
        def _add_children(descendants, table):
            children = list(self.graph.predecessors(table))
            if len(children) > 0:
                descendants.update(children)
                for child in children:
                    _add_children(descendants, child)

        descendants = set()
        _add_children(descendants, table)

        return list(descendants)

    def list_tables_parents_before_children(self) -> List[str]:
        """
        Returns a list of all tables with the guarantee that a parent table
        appears before any of its children. No other guarantees about order
        are made, e.g. the following (and others) are all valid outputs:
        [p1, p2, c1, c2] or [p2, c2, p1, c1] or [p2, p1, c1, c2] etc.
        """
        return list(reversed(list(topological_sort(self.graph))))

    def get_primary_key(self, table: str) -> List[str]:
        try:
            return self.graph.nodes[table]["metadata"].primary_key
        except KeyError:
            if table in self.relational_jsons:
                return self.relational_jsons[table].original_primary_key
            else:
                raise MultiTableException(f"Unrecognized table: `{table}`")

    def get_table_data(
        self, table: str, usecols: Optional[set[str]] = None
    ) -> pd.DataFrame:
        usecols = usecols or self.get_table_columns(table)
        try:
            return self.graph.nodes[table]["metadata"].data[list(usecols)]
        except KeyError:
            if table in self.relational_jsons:
                if (df := self.relational_jsons[table].original_data) is None:
                    raise MultiTableException("Original data with JSON is lost.")
                return df
            else:
                raise MultiTableException(f"Unrecognized table: `{table}`")

    def get_table_columns(self, table: str) -> set[str]:
        try:
            return self.graph.nodes[table]["metadata"].columns
        except KeyError:
            if table in self.relational_jsons:
                return set(self.relational_jsons[table].original_columns)
            else:
                raise MultiTableException(f"Unrecognized table: `{table}`")

    def get_safe_ancestral_seed_columns(self, table: str) -> set[str]:
        safe_columns = self.graph.nodes[table]["metadata"].safe_ancestral_seed_columns
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

        self.graph.nodes[table]["metadata"].safe_ancestral_seed_columns = cols
        return cols

    def _clear_safe_ancestral_seed_columns(self, table: str) -> None:
        self.graph.nodes[table]["metadata"].safe_ancestral_seed_columns = None

    def _get_table_in_graph(self, table: str) -> str:
        if table in self.relational_jsons:
            table = self.relational_jsons[table].root_table_name
        return table

    def get_foreign_keys(
        self, table: str, rename_invented_tables: bool = False
    ) -> List[ForeignKey]:
        def _rename_invented(fk: ForeignKey) -> ForeignKey:
            table_name = fk.table_name
            parent_table_name = fk.parent_table_name
            if self._is_invented(table_name):
                table_name = self.graph.nodes[table_name][
                    "metadata"
                ].invented_table_metadata.original_table_name
            if self._is_invented(parent_table_name):
                parent_table_name = self.graph.nodes[parent_table_name][
                    "metadata"
                ].invented_table_metadata.original_table_name
            return replace(
                fk, table_name=table_name, parent_table_name=parent_table_name
            )

        table = self._get_table_in_graph(table)
        foreign_keys = []
        for parent in self.get_parents(table):
            fks = self.graph.edges[table, parent]["via"]
            foreign_keys.extend(fks)

        if rename_invented_tables:
            return [_rename_invented(fk) for fk in foreign_keys]
        else:
            return foreign_keys

    def get_all_key_columns(self, table: str) -> List[str]:
        all_key_cols = []
        all_key_cols.extend(self.get_primary_key(table))
        for fk in self.get_foreign_keys(table):
            all_key_cols.extend(fk.columns)
        return all_key_cols

    def debug_summary(self) -> Dict[str, Any]:
        max_depth = dag_longest_path_length(self.graph)
        public_table_count = len(self.list_all_tables(Scope.PUBLIC))
        invented_table_count = len(self.list_all_tables(Scope.INVENTED))

        all_tables = self.list_all_tables(Scope.ALL)
        total_foreign_key_count = 0
        tables = {}
        for table in all_tables:
            this_table_foreign_key_count = 0
            foreign_keys = []
            fk_lookup_table_name = self._get_table_in_graph(table)
            for key in self.get_foreign_keys(fk_lookup_table_name):
                total_foreign_key_count = total_foreign_key_count + 1
                this_table_foreign_key_count = this_table_foreign_key_count + 1
                foreign_keys.append(
                    {
                        "columns": key.columns,
                        "parent_table_name": key.parent_table_name,
                        "parent_columns": key.parent_columns,
                    }
                )
            tables[table] = {
                "column_count": len(self.get_table_columns(table)),
                "primary_key": self.get_primary_key(table),
                "foreign_key_count": this_table_foreign_key_count,
                "foreign_keys": foreign_keys,
                "is_invented_table": self._is_invented(table),
            }
        return {
            "foreign_key_count": total_foreign_key_count,
            "max_depth": max_depth,
            "tables": tables,
            "public_table_count": public_table_count,
            "invented_table_count": invented_table_count,
        }


def skip_table(
    table: str, only: Optional[list[str]], ignore: Optional[list[str]]
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
