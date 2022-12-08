from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx
import pandas as pd


class MultiTableException(Exception):
    pass


@dataclass
class ForeignKey:
    column_name: str
    parent_column_name: str
    parent_table_name: str


class RelationalData:
    def __init__(self):
        self.graph = networkx.DiGraph()
        self.lineage_column_delimiter = "|"

    def add_table(
        self, table: str, primary_key: Optional[str], data: pd.DataFrame
    ) -> None:
        self.graph.add_node(table, primary_key=primary_key, data=data)

    def add_foreign_key(self, foreign_key: str, referencing: str) -> None:
        """Format of both str arguments should be `table_name.column_name`"""
        fk_table, fk_column = foreign_key.split(".")
        referenced_table, referenced_column = referencing.split(".")
        self.graph.add_edge(fk_table, referenced_table)
        edge = self.graph.edges[fk_table, referenced_table]
        via = edge.get("via", [])
        via.append(
            ForeignKey(
                column_name=fk_column,
                parent_column_name=referenced_column,
                parent_table_name=referenced_table,
            )
        )
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

    def get_primary_key(self, table: str) -> Optional[str]:
        return self.graph.nodes[table]["primary_key"]

    def get_table_data(self, table: str) -> pd.DataFrame:
        return self.graph.nodes[table]["data"]

    def get_foreign_keys(self, table: str) -> List[ForeignKey]:
        foreign_keys = []
        for parent in self.get_parents(table):
            fks = self.graph.edges[table, parent]["via"]
            foreign_keys.extend(fks)
        return foreign_keys

    def get_table_data_with_ancestors(
        self, table: str, tableset: Optional[Dict[str, pd.DataFrame]] = None
    ) -> pd.DataFrame:
        """
        Returns a data frame with all ancestral data joined to each record.
        Column names are modified to the format `LINAGE|COLUMN_NAME`.
        Lineage begins with `self` for the supplied `table`, and as older
        generations are joined, the foreign keys to those generations are appended,
        separated by periods.

        If `tableset` is provided, use it in place of the source data in `self.graph`.
        """
        lineage = "self"
        if tableset is not None:
            df = tableset[table]
        else:
            df = self.get_table_data(table)
        df = df.add_prefix(f"{lineage}{self.lineage_column_delimiter}")
        return _join_parents(df, table, lineage, self, tableset)

    def drop_ancestral_data(self, df: pd.DataFrame) -> pd.DataFrame:
        delim = self.lineage_column_delimiter
        root_columns = [col for col in df.columns if col.startswith(f"self{delim}")]
        mapper = {col: col.removeprefix(f"self{delim}") for col in root_columns}
        return df[root_columns].rename(columns=mapper)

    def build_seed_data_for_table(
        self, table: str, ancestor_data: Dict[str, pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        foreign_keys = self.get_foreign_keys(table)
        # TODO: check and raise if ancestor_data is missing any parents?

        if len(foreign_keys) == 0:
            return None
        elif len(foreign_keys) == 1:
            foreign_key = foreign_keys[0]
            parent_df = ancestor_data[foreign_key.parent_table_name]
            mapper = {
                col: col.replace("self", f"self.{foreign_key.column_name}", 1)
                for col in parent_df.columns
            }
            return parent_df.rename(columns=mapper)
        else:
            # TODO: determine what seed should look like when table has multiple FKs
            return None

    def as_dict(self, out_dir: str) -> Dict[str, Any]:
        d = {"tables": {}, "foreign_keys": []}
        for table in self.list_all_tables():
            d["tables"][table] = {
                "primary_key": self.get_primary_key(table),
                "csv_path": f"{out_dir}/{table}.csv",
            }
            keys = [
                (
                    f"{table}.{key.column_name}",
                    f"{key.parent_table_name}.{key.parent_column_name}",
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
            relational_data.add_table(table_name, primary_key, data)
        for foreign_key_tuple in d["foreign_keys"]:
            foreign_key, referencing = foreign_key_tuple
            relational_data.add_foreign_key(foreign_key, referencing)

        return relational_data


def _join_parents(
    df: pd.DataFrame,
    table: str,
    lineage: str,
    relational_data: RelationalData,
    tableset: Optional[Dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    delim = relational_data.lineage_column_delimiter
    for foreign_key in relational_data.get_foreign_keys(table):
        next_lineage = f"{lineage}.{foreign_key.column_name}"

        parent_table_name = foreign_key.parent_table_name
        if tableset is not None:
            parent_data = tableset[parent_table_name]
        else:
            parent_data = relational_data.get_table_data(parent_table_name)
        parent_data = parent_data.add_prefix(f"{next_lineage}{delim}")

        df = df.merge(
            parent_data,
            how="left",
            left_on=f"{lineage}{delim}{foreign_key.column_name}",
            right_on=f"{next_lineage}{delim}{foreign_key.parent_column_name}",
        )

        df = _join_parents(
            df, parent_table_name, next_lineage, relational_data, tableset
        )
    return df
