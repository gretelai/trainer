from typing import Any, Dict, List

import pandas as pd

from gretel_trainer.relational.core import RelationalData


class SingleTableStrategy:
    def prepare_training_data(
        self, table_name: str, rel_data: RelationalData
    ) -> pd.DataFrame:
        """
        Returns the source table data with primary and foreign keys removed
        """
        data = rel_data.get_table_data(table_name)
        columns_to_drop = []

        primary_key = rel_data.get_primary_key(table_name)
        if primary_key is not None:
            columns_to_drop.append(primary_key)
        foreign_keys = rel_data.get_foreign_keys(table_name)
        columns_to_drop.extend(
            [foreign_key.column_name for foreign_key in foreign_keys]
        )

        return data.drop(columns=columns_to_drop)

    def tables_to_retrain(
        self, tables: List[str], rel_data: RelationalData
    ) -> List[str]:
        """
        Returns the provided tables requested to retrain, unaltered.
        """
        return tables

    def get_generation_jobs(
        self, table: str, rel_data: RelationalData, record_size_ratio: float, output_tables: Dict[str, pd.DataFrame]
    ) -> List[Dict[str, Any]]:
        source_data_size = len(rel_data.get_table_data(table))
        synth_size = int(source_data_size * record_size_ratio)
        return [
            {"num_records": synth_size}
        ]
