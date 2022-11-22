from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx
import pandas as pd


@dataclass
class ForeignKey:
    column_name: str
    parent_column_name: str
    parent_table_name: str


class RelationalData:
    def __init__(self):
        self.graph = networkx.DiGraph()
        self.lineage_column_delimiter = "|"

    def add_table(self, table: str, primary_key: Optional[str], data: pd.DataFrame) -> None:
        self.graph.add_node(table, primary_key=primary_key, data=data)

    def add_foreign_key(self, foreign_key: str, referencing: str) -> None:
        """Format of both str arguments should be `table_name.column_name`"""
        fk_table, fk_column = foreign_key.split(".")
        referenced_table, referenced_column = referencing.split(".")
        self.graph.add_edge(fk_table, referenced_table)
        edge = self.graph.edges[fk_table, referenced_table]
        via = edge.get("via", [])
        via.append(ForeignKey(
            column_name=fk_column,
            parent_column_name=referenced_column,
            parent_table_name=referenced_table,
        ))
        edge["via"] = via

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

    def get_table_data_with_ancestors(self, table: str) -> pd.DataFrame:
        """
        Returns a data frame with all ancestral data joined to each record.
        Column names are modified to the format `LINAGE|COLUMN_NAME`.
        Lineage begins with `self` for the supplied `table`, and as older
        generations are joined, the foreign keys to those generations are appended,
        separated by periods.
        """
        lineage = "self"
        df = self.get_table_data(table)
        df = df.add_prefix(f"{lineage}{self.lineage_column_delimiter}")
        return _join_parents(df, table, lineage, self)

    def drop_ancestral_data(self, df: pd.DataFrame) -> pd.DataFrame:
        delim = self.lineage_column_delimiter
        root_columns = [col for col in df.columns if col.startswith(f"self{delim}")]
        mapper = { col: col.removeprefix(f"self{delim}") for col in root_columns }
        return df[root_columns].rename(columns=mapper)

    def build_seed_data_for_table(self, table: str, ancestor_data: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
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


def _join_parents(
    df: pd.DataFrame,
    table: str,
    lineage: str,
    relational_data: RelationalData
) -> pd.DataFrame:
    delim = relational_data.lineage_column_delimiter
    for foreign_key in relational_data.get_foreign_keys(table):
        next_lineage = f"{lineage}.{foreign_key.column_name}"

        parent_table_name = foreign_key.parent_table_name
        parent_data = relational_data.get_table_data(parent_table_name)
        parent_data = parent_data.add_prefix(f"{next_lineage}{delim}")

        df = df.merge(
            parent_data,
            how="left",
            left_on=f"{lineage}{delim}{foreign_key.column_name}",
            right_on=f"{next_lineage}{delim}{foreign_key.parent_column_name}",
        )

        df = _join_parents(df, parent_table_name, next_lineage, relational_data)
    return df


class TableProgress:
    def __init__(self, relational_data: RelationalData):
        self.relational_data = relational_data
        self.table_statuses = {
            table: False
            for table in relational_data.list_all_tables()
        }

    def mark_complete(self, table: str) -> None:
        self.table_statuses[table] = True

    def ready(self) -> List[str]:
        ready = []
        for table, processed in self.table_statuses.items():
            if processed:
                continue

            parents = self.relational_data.get_parents(table)
            if len(parents) == 0:
                ready.append(table)
                continue

            if all([self.table_statuses[parent] for parent in parents]):
                ready.append(table)

        return ready
