from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx
import pandas as pd
from gretel_client.evaluation.quality_report import QualityReport
from networkx.algorithms.dag import dag_longest_path_length

logger = logging.getLogger(__name__)


class MultiTableException(Exception):
    pass


@dataclass
class TableEvaluation:
    individual_sqs: int
    cross_table_sqs: int


@dataclass
class ForeignKey:
    column_name: str
    parent_column_name: str
    parent_table_name: str


def get_sqs_via_evaluate(data_source: pd.DataFrame, ref_data: pd.DataFrame) -> int:
    report = QualityReport(data_source=data_source, ref_data=ref_data)
    report.run()
    return report.peek()["score"]


class RelationalData:
    def __init__(self):
        self.graph = networkx.DiGraph()
        self._lineage_column_delimiter = "|"
        self._generation_delimiter = "."

    def add_table(
        self, table: str, primary_key: Optional[str], data: pd.DataFrame
    ) -> None:
        self.graph.add_node(table, primary_key=primary_key, data=data)

    def add_foreign_key(self, foreign_key: str, referencing: str) -> None:
        """Format of both str arguments should be `table_name.column_name`"""
        known_tables = self.list_all_tables()
        fk_table, fk_column = foreign_key.split(".")
        referenced_table, referenced_column = referencing.split(".")

        abort = False
        if fk_table not in known_tables:
            logger.warning(f"Unrecognized table name: `{fk_table}`")
            abort = True
        if referenced_table not in known_tables:
            logger.warning(f"Unrecognized table name: `{referenced_table}`")
            abort = True

        if abort:
            return None

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

    def get_ancestral_foreign_key_maps(self, table: str) -> List[Tuple[str, str]]:
        def _ancestral_fk_map(fk: ForeignKey) -> Tuple[str, str]:
            gen_delim = self._generation_delimiter
            lineage_delim = self._lineage_column_delimiter
            fk_col = fk.column_name
            ref_col = fk.parent_column_name

            ancestral_foreign_key = f"self{lineage_delim}{fk_col}"
            ancestral_referenced_col = (
                f"self{gen_delim}{fk_col}{lineage_delim}{ref_col}"
            )

            return (ancestral_foreign_key, ancestral_referenced_col)

        return [_ancestral_fk_map(fk) for fk in self.get_foreign_keys(table)]

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
        df = df.add_prefix(f"{lineage}{self._lineage_column_delimiter}")
        return _join_parents(df, table, lineage, self, tableset)

    def list_multigenerational_keys(self, table: str) -> List[str]:
        """
        Returns a list of multigenerational column names (i.e. including lineage)
        that are primary or foreign keys on the source tables.
        """

        def _add_multigenerational_keys(keys: List[str], lineage: str, table_name: str):
            primary_key = self.get_primary_key(table_name)
            if primary_key is not None:
                keys.append(f"{lineage}{self._lineage_column_delimiter}{primary_key}")

            foreign_keys = self.get_foreign_keys(table_name)
            keys.extend(
                [
                    f"{lineage}{self._lineage_column_delimiter}{foreign_key.column_name}"
                    for foreign_key in foreign_keys
                ]
            )

            for foreign_key in foreign_keys:
                next_lineage = (
                    f"{lineage}{self._generation_delimiter}{foreign_key.column_name}"
                )
                parent_table_name = foreign_key.parent_table_name
                _add_multigenerational_keys(keys, next_lineage, parent_table_name)

        keys = []
        _add_multigenerational_keys(keys, "self", table)
        return keys

    def is_ancestral_column(self, column: str) -> bool:
        """
        Returns True if the provided column name corresponds to an elder-generation ancestor.
        """
        regex_string = rf"\{self._generation_delimiter}[^\{self._lineage_column_delimiter}]+\{self._lineage_column_delimiter}"
        regex = re.compile(regex_string)
        return bool(regex.search(column))

    def drop_ancestral_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops ancestral columns from the given dataframe and removes the lineage prefix
        from the remaining columns, restoring them to their original source names.
        """
        delim = self._lineage_column_delimiter
        root_columns = [col for col in df.columns if col.startswith(f"self{delim}")]
        mapper = {col: col.removeprefix(f"self{delim}") for col in root_columns}
        return df[root_columns].rename(columns=mapper)

    def build_seed_data_for_table(
        self, table: str, ancestor_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Returns a multigenerational dataframe composed exclusively of ancestral columns;
        columns from the provided table are excluded. If ancestor_data is provided, will
        use that tableset; otherwise uses source data. The ancestor_data dict MAY include
        a key/value pair for for the provided table, but it is not necessary because those
        columns are not included in the output dataframe.

        Returns None if table has no parents.
        """
        if len(self.get_parents(table)) == 0:
            return None
        else:
            if ancestor_data is not None:
                # TODO: check and raise if ancestor_data is missing any parents?

                # Ensure provided data is not multigenerational; columns should match source
                # TODO: do we need to be defensive here and explicitly check for lineage prefixes?
                for name, data in ancestor_data.items():
                    ancestor_data[name] = self.drop_ancestral_data(data)

                # Data from supplied `table` must be present for the call to `get_table_data_with_ancestors`,
                # but those columns are not included in the output so it's OK to add source data to an
                # otherwise synthetic tableset
                if ancestor_data.get(table) is None:
                    ancestor_data.update({table: self.get_table_data(table)})

            df = self.get_table_data_with_ancestors(table, ancestor_data)
            ancestral_columns = [
                col for col in df.columns if self.is_ancestral_column(col)
            ]
            return df.filter(ancestral_columns)

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
                        "column_name": key.column_name,
                        "parent_column_name": key.parent_column_name,
                        "parent_table_name": key.parent_table_name,
                    }
                )
            tables[table] = {
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
    delim = relational_data._lineage_column_delimiter
    for foreign_key in relational_data.get_foreign_keys(table):
        next_lineage = (
            f"{lineage}{relational_data._generation_delimiter}{foreign_key.column_name}"
        )

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
