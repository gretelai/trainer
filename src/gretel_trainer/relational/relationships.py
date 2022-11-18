from __future__ import annotations
from typing import List, Tuple

import networkx as nx


class Relationships:
    def __init__(self):
        self.g = nx.DiGraph()

    def all_tables(self) -> List[str]:
        return list(self.g.nodes)

    def add(self, fk: Tuple[str, str], referencing: Tuple[str, str]) -> Relationships:
        fk_table, fk_column = fk
        pk_table, pk_column = referencing

        self.g.add_edge(
            fk_table,
            pk_table,
            fk={"table": fk_table, "column": fk_column},
            pk={"table": pk_table, "column": pk_column},
        )
        return self

    def get_parents(self, table: str) -> List[str]:
        return list(self.g.successors(table))


class TableProgress:
    def __init__(self, relationships: Relationships):
        self.relationships = relationships
        self.table_statuses = {
            table: False
            for table in relationships.all_tables()
        }

    def mark_complete(self, table: str) -> None:
        self.table_statuses[table] = True

    def ready(self) -> List[str]:
        ready = []
        for table, processed in self.table_statuses.items():
            if processed:
                continue

            parents = self.relationships.get_parents(table)
            if len(parents) == 0:
                ready.append(table)
                continue

            if all([self.table_statuses[parent] for parent in parents]):
                ready.append(table)

        return ready
