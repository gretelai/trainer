from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, overload

import networkx
import pandas as pd
from networkx.algorithms.dag import dag_longest_path_length
from typing_extensions import Literal

logger = logging.getLogger(__name__)

_SQS = "synthetic_data_quality_score"
_PPL = "privacy_protection_level"
_SCORE = "score"
_GRADE = "grade"


class MultiTableException(Exception):
    pass


GretelModelConfig = Union[str, Path, Dict]


@dataclass
class TableEvaluation:
    cross_table_report_json: Optional[Dict] = field(default=None, repr=False)
    individual_report_json: Optional[Dict] = field(default=None, repr=False)

    def is_complete(self) -> bool:
        return (
            self.cross_table_report_json is not None
            and self.cross_table_sqs is not None
            and self.individual_report_json is not None
            and self.individual_sqs is not None
        )

    @overload
    def _field_from_json(
        self, report_json: Optional[dict], entry: str, field: Literal["score"]
    ) -> Optional[int]:
        ...

    @overload
    def _field_from_json(
        self, report_json: Optional[dict], entry: str, field: Literal["grade"]
    ) -> Optional[str]:
        ...

    def _field_from_json(
        self, report_json: Optional[dict], entry: str, field: str
    ) -> Optional[Union[int, str]]:
        if report_json is None:
            return None
        else:
            return report_json.get(entry, {}).get(field)

    @property
    def cross_table_sqs(self) -> Optional[int]:
        return self._field_from_json(self.cross_table_report_json, _SQS, _SCORE)

    @property
    def cross_table_sqs_grade(self) -> Optional[str]:
        return self._field_from_json(self.cross_table_report_json, _SQS, _GRADE)

    @property
    def cross_table_ppl(self) -> Optional[int]:
        return self._field_from_json(self.cross_table_report_json, _PPL, _SCORE)

    @property
    def cross_table_ppl_grade(self) -> Optional[str]:
        return self._field_from_json(self.cross_table_report_json, _PPL, _GRADE)

    @property
    def individual_sqs(self) -> Optional[int]:
        return self._field_from_json(self.individual_report_json, _SQS, _SCORE)

    @property
    def individual_sqs_grade(self) -> Optional[str]:
        return self._field_from_json(self.individual_report_json, _SQS, _GRADE)

    @property
    def individual_ppl(self) -> Optional[int]:
        return self._field_from_json(self.individual_report_json, _PPL, _SCORE)

    @property
    def individual_ppl_grade(self) -> Optional[str]:
        return self._field_from_json(self.individual_report_json, _PPL, _GRADE)

    def __repr__(self) -> str:
        d = {}
        if self.cross_table_report_json is not None:
            d["cross_table"] = {
                "sqs": {
                    "score": self.cross_table_sqs,
                    "grade": self.cross_table_sqs_grade,
                },
                "ppl": {
                    "score": self.cross_table_ppl,
                    "grade": self.cross_table_ppl_grade,
                },
            }
        if self.individual_report_json is not None:
            d["individual"] = {
                "sqs": {
                    "score": self.individual_sqs,
                    "grade": self.individual_sqs_grade,
                },
                "ppl": {
                    "score": self.individual_ppl,
                    "grade": self.individual_ppl_grade,
                },
            }
        return json.dumps(d)


@dataclass
class ForeignKey:
    table_name: str
    columns: List[str]
    parent_table_name: str
    parent_columns: List[str]


UserFriendlyPrimaryKeyT = Optional[Union[str, List[str]]]
UserFriendlyForeignKeyT = Tuple[str, Union[str, List[str]]]


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
        self.graph.add_node(name, primary_key=primary_key, data=data)

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

        known_columns = self.get_table_data(table).columns
        for col in primary_key:
            if col not in known_columns:
                raise MultiTableException(f"Unrecognized column name: `{primary_key}`")

        self.graph.nodes[table]["primary_key"] = primary_key

    def _format_key_column(self, key: Optional[Union[str, List[str]]]) -> List[str]:
        if key is None:
            return []
        elif isinstance(key, str):
            return [key]
        else:
            return key

    def add_foreign_key(
        self,
        *,
        foreign_key: UserFriendlyForeignKeyT,
        referencing: UserFriendlyForeignKeyT,
    ) -> None:
        """
        Add a foreign key relationship between two tables.
        The first element of each tuple argument is a table name.
        The second element can be a single column (most common) or a list of columns (for composite keys).
        """
        known_tables = self.list_all_tables()
        fk_table, fk_columns = foreign_key
        fk_columns = self._format_key_column(fk_columns)
        referenced_table, referenced_columns = referencing
        referenced_columns = self._format_key_column(referenced_columns)

        abort = False
        if fk_table not in known_tables:
            logger.warning(f"Unrecognized table name: `{fk_table}`")
            abort = True
        if referenced_table not in known_tables:
            logger.warning(f"Unrecognized table name: `{referenced_table}`")
            abort = True
        if len(fk_columns) != len(referenced_columns):
            logger.warning(
                "Foreign key and referenced columns must be of the same length"
            )
            abort = True

        if abort:
            return None

        self.graph.add_edge(fk_table, referenced_table)
        edge = self.graph.edges[fk_table, referenced_table]
        via = edge.get("via", [])
        via.append(
            ForeignKey(
                table_name=fk_table,
                columns=self._format_key_column(fk_columns),
                parent_table_name=referenced_table,
                parent_columns=self._format_key_column(referenced_columns),
            )
        )
        edge["via"] = via

    def remove_foreign_key(self, foreign_key: UserFriendlyForeignKeyT) -> None:
        """
        Remove an existing foreign key.
        The first element of the tuple argument is a table name.
        The second element can be a single column (most common) or a list of columns (for composite keys).
        """
        fk_table, fk_columns = foreign_key
        fk_columns = self._format_key_column(fk_columns)

        if fk_table not in self.list_all_tables():
            raise MultiTableException(f"Unrecognized table name: `{fk_table}`")

        if any(col not in self.get_table_data(fk_table).columns for col in fk_columns):
            raise MultiTableException(f"Column does not exist on table `{fk_table}`")

        key_to_remove = None
        for fk in self.get_foreign_keys(fk_table):
            if fk.columns == fk_columns:
                key_to_remove = fk

        if key_to_remove is None:
            raise MultiTableException(
                f"`{fk_columns}` on table `{fk_table}` is not a foreign key"
            )

        edge = self.graph.edges[fk_table, key_to_remove.parent_table_name]
        via = edge.get("via")
        via.remove(key_to_remove)
        if len(via) == 0:
            self.graph.remove_edge(fk_table, key_to_remove.parent_table_name)
        else:
            edge["via"] = via

    def update_table_data(self, table: str, data: pd.DataFrame) -> None:
        try:
            self.graph.nodes[table]["data"] = data
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

    def get_primary_key(self, table: str) -> List[str]:
        return self.graph.nodes[table]["primary_key"]

    def get_table_data(self, table: str) -> pd.DataFrame:
        return self.graph.nodes[table]["data"]

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
                (
                    (table, key.columns),
                    (key.parent_table_name, key.parent_columns),
                )
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
        for foreign_key_tuple in d["foreign_keys"]:
            foreign_key, referencing = foreign_key_tuple
            relational_data.add_foreign_key(
                foreign_key=foreign_key, referencing=referencing
            )

        return relational_data
