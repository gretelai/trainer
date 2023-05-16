from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import networkx
import pandas as pd
from networkx.algorithms.dag import dag_longest_path_length, topological_sort
from pandas.api.types import is_string_dtype

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


@dataclass
class TableMetadata:
    primary_key: list[str]
    data: pd.DataFrame
    columns: set[str]
    safe_ancestral_seed_columns: Optional[set[str]] = None


class RelationalData:
    def __init__(self):
        self.graph = networkx.DiGraph()

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
        """
        primary_key = self._format_key_column(primary_key)
        metadata = TableMetadata(
            primary_key=primary_key, data=data, columns=set(data.columns)
        )
        self.graph.add_node(name, metadata=metadata)

    def set_primary_key(
        self, *, table: str, primary_key: UserFriendlyPrimaryKeyT
    ) -> None:
        """
        (Re)set the primary key on an existing table.
        If the table does not yet exist in the instance's collection, add it via `add_table`.
        """
        if table not in self.list_all_tables():
            raise MultiTableException(f"Unrecognized table name: `{table}`")

        primary_key = self._format_key_column(primary_key)

        known_columns = self.get_table_columns(table)
        for col in primary_key:
            if col not in known_columns:
                raise MultiTableException(f"Unrecognized column name: `{primary_key}`")

        self.graph.nodes[table]["metadata"].primary_key = primary_key
        self._clear_safe_ancestral_seed_columns(table)

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
        known_tables = self.list_all_tables()

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
        if table not in self.list_all_tables():
            raise MultiTableException(f"Unrecognized table name: `{table}`")

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
        try:
            self.graph.nodes[table]["metadata"].data = data
            self.graph.nodes[table]["metadata"].columns = set(data.columns)
            self._clear_safe_ancestral_seed_columns(table)
        except KeyError:
            raise MultiTableException(
                f"Unrecognized table name: {table}. If this is a new table to add, use `add_table`."
            )

    def list_all_tables(self) -> List[str]:
        return list(self.graph.nodes)

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
        return self.graph.nodes[table]["metadata"].primary_key

    def get_table_data(
        self, table: str, usecols: Optional[set[str]] = None
    ) -> pd.DataFrame:
        usecols = usecols or self.get_table_columns(table)
        return self.graph.nodes[table]["metadata"].data[list(usecols)]

    def get_table_columns(self, table: str) -> set[str]:
        return self.graph.nodes[table]["metadata"].columns

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

    def get_foreign_keys(self, table: str) -> List[ForeignKey]:
        foreign_keys = []
        for parent in self.get_parents(table):
            fks = self.graph.edges[table, parent]["via"]
            foreign_keys.extend(fks)
        return foreign_keys

    def get_all_key_columns(self, table: str) -> List[str]:
        all_key_cols = []
        all_key_cols.extend(self.get_primary_key(table))
        for fk in self.get_foreign_keys(table):
            all_key_cols.extend(fk.columns)
        return all_key_cols

    def debug_summary(self) -> Dict[str, Any]:
        max_depth = dag_longest_path_length(self.graph)
        all_tables = self.list_all_tables()
        table_count = len(all_tables)
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
            tables[table] = {
                "column_count": len(self.get_table_data(table).columns),
                "primary_key": self.get_primary_key(table),
                "foreign_key_count": this_table_foreign_key_count,
                "foreign_keys": foreign_keys,
            }
        return {
            "foreign_key_count": total_foreign_key_count,
            "max_depth": max_depth,
            "table_count": table_count,
            "tables": tables,
        }

    def as_dict(self, out_dir: str) -> Dict[str, Any]:
        d = {"tables": {}, "foreign_keys": []}
        for table in self.list_all_tables():
            d["tables"][table] = {
                "primary_key": self.get_primary_key(table),
                "csv_path": f"{out_dir}/{table}.csv",
            }
            keys = [
                {
                    "table": table,
                    "constrained_columns": key.columns,
                    "referred_table": key.parent_table_name,
                    "referred_columns": key.parent_columns,
                }
                for key in self.get_foreign_keys(table)
            ]
            d["foreign_keys"].extend(keys)
        return d

    def to_filesystem(self, out_dir: str) -> str:
        d = self.as_dict(out_dir)
        for table_name, details in d["tables"].items():
            self.get_table_data(table_name).to_csv(details["csv_path"], index=False)
        metadata_path = f"{out_dir}/metadata.json"
        with open(metadata_path, "w") as metadata_file:
            json.dump(d, metadata_file)
        return metadata_path

    @classmethod
    def from_filesystem(cls, metadata_filepath: str) -> RelationalData:
        with open(metadata_filepath, "r") as metadata_file:
            d = json.load(metadata_file)
        relational_data = RelationalData()

        for table_name, details in d["tables"].items():
            primary_key = details["primary_key"]
            data = pd.read_csv(details["csv_path"])
            relational_data.add_table(
                name=table_name, primary_key=primary_key, data=data
            )
        for foreign_key in d["foreign_keys"]:
            relational_data.add_foreign_key_constraint(
                table=foreign_key["table"],
                constrained_columns=foreign_key["constrained_columns"],
                referred_table=foreign_key["referred_table"],
                referred_columns=foreign_key["referred_columns"],
            )

        return relational_data


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
