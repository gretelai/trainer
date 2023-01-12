from typing import Any, Dict, List, Optional

import pandas as pd

from gretel_trainer.relational.core import (
    RelationalData,
    TableEvaluation,
    get_sqs_via_evaluate,
)


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

    def get_seed_fields(
        self, table_name: str, rel_data: RelationalData
    ) -> Optional[List[str]]:
        return None

    def tables_to_retrain(
        self, tables: List[str], rel_data: RelationalData
    ) -> List[str]:
        """
        Returns the provided tables requested to retrain, unaltered.
        """
        return tables

    def ready_to_generate(
        self,
        rel_data: RelationalData,
        in_progress: List[str],
        finished: List[str],
    ) -> List[str]:
        """
        All tables are immediately ready for generation. Once they are
        at least in progress, they are no longer ready.
        """
        return [
            table
            for table in rel_data.list_all_tables()
            if table not in in_progress and table not in finished
        ]

    def get_generation_job(
        self,
        table: str,
        rel_data: RelationalData,
        record_size_ratio: float,
        output_tables: Dict[str, pd.DataFrame],
    ) -> Dict[str, Any]:
        """
        Returns kwargs for a record handler job requesting an output record
        count based on the initial table data size and the record size ratio.
        """
        source_data_size = len(rel_data.get_table_data(table))
        synth_size = int(source_data_size * record_size_ratio)
        return {"params": {"num_records": synth_size}}

    def get_generation_jobs(
        self,
        table: str,
        rel_data: RelationalData,
        record_size_ratio: float,
        output_tables: Dict[str, pd.DataFrame],
    ) -> List[Dict[str, Any]]:
        """
        Returns one job requesting an output record count based on
        the initial table data size and the record size ratio.
        """
        source_data_size = len(rel_data.get_table_data(table))
        synth_size = int(source_data_size * record_size_ratio)
        return [{"num_records": synth_size}]

    def collect_generation_results(
        self, results: List[pd.DataFrame], table_name: str, rel_data: RelationalData
    ) -> pd.DataFrame:
        """
        Concatenates all results, which should just be a list of one element.
        """
        return pd.concat(results)

    def evaluate(
        self,
        table: str,
        rel_data: RelationalData,
        model_score: int,
        synthetic_tables: Dict[str, pd.DataFrame],
    ) -> TableEvaluation:
        ancestral_synthetic_data = rel_data.get_table_data_with_ancestors(
            table, synthetic_tables
        )
        ancestral_reference_data = rel_data.get_table_data_with_ancestors(table)
        ancestral_score = get_sqs_via_evaluate(
            ancestral_synthetic_data, ancestral_reference_data
        )

        return TableEvaluation(
            individual_sqs=model_score, ancestral_sqs=ancestral_score
        )
