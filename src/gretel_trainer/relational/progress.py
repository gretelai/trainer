from typing import List

from gretel_trainer.relational.core import RelationalData


class TableProgress:
    def __init__(self, relational_data: RelationalData):
        self.relational_data = relational_data
        self.table_statuses = {
            table: False for table in relational_data.list_all_tables()
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
