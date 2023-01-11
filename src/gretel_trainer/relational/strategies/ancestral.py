from collections import defaultdict
from typing import Any, Dict, List, Optional

import pandas as pd
from pandas.api.types import is_string_dtype

from gretel_trainer.relational.core import (
    MultiTableException,
    RelationalData,
    TableEvaluation,
    get_sqs_via_evaluate,
)


class AncestralStrategy:
    def __init__(self, model_type: str = "amplify"):
        if model_type not in ("amplify", "lstm"):
            raise MultiTableException(
                f"Unsupported model type: {model_type}. Supported model types are `Amplify` and `LSTM`."
            )
        self._model_type = model_type

    def prepare_training_data(
        self, table_name: str, rel_data: RelationalData
    ) -> pd.DataFrame:
        """
        Returns table data with all ancestor fields added,
        minus any highly-unique categorical fields from ancestors.
        """
        data = rel_data.get_table_data_with_ancestors(table_name)
        columns_to_drop = []

        for column in data.columns:
            if rel_data.is_ancestral_column(column) and _is_highly_unique_categorical(
                column, data
            ):
                columns_to_drop.append(column)

        return data.drop(columns=columns_to_drop)

    def get_seed_fields(
        self, table_name: str, rel_data: RelationalData
    ) -> Optional[List[str]]:
        seed_data = rel_data.build_seed_data_for_table(table_name)
        if seed_data is None:
            return None
        else:
            return [
                col for col in seed_data.columns if rel_data.is_ancestral_column(col)
            ]

    def tables_to_retrain(
        self, tables: List[str], rel_data: RelationalData
    ) -> List[str]:
        """
        Given a set of tables requested to retrain, returns those tables with all their
        descendants, because those descendant tables were trained with data from their
        parents appended.
        """
        retrain = set(tables)
        for table in tables:
            retrain.update(rel_data.get_descendants(table))
        return list(retrain)

    def ready_to_generate(
        self,
        rel_data: RelationalData,
        in_progress: List[str],
        finished: List[str],
    ) -> List[str]:
        """
        Tables with no parents are immediately ready for generation.
        Tables with parents are ready once their parents are finished.
        All tables are no longer considered ready once they are at least in progress.
        """
        ready = []

        for table in rel_data.list_all_tables():
            if table in in_progress or table in finished:
                continue

            parents = rel_data.get_parents(table)
            if len(parents) == 0:
                ready.append(table)
            elif all([parent in finished for parent in parents]):
                ready.append(table)

        return ready

    def get_generation_jobs(
        self,
        table: str,
        rel_data: RelationalData,
        record_size_ratio: float,
        output_tables: Dict[str, pd.DataFrame],
    ) -> List[Dict[str, Any]]:
        """
        If the table does not have any parents, returns a single job requesting an output
        record count based on the initial table data size and the record size ratio.

        If the table does have parents, builds a seed dataframe to use in generation.
        """
        if len(rel_data.get_parents(table)) == 0:
            requested_synth_count = (
                len(rel_data.get_table_data(table)) * record_size_ratio
            )
            return [{"num_records": requested_synth_count}]
        else:
            seed_df = rel_data.build_seed_data_for_table(table, output_tables)
            return [{"seed_df": seed_df}]

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
        individual_synthetic_data = synthetic_tables[table]
        individual_reference_data = rel_data.get_table_data(table)
        individual_score = get_sqs_via_evaluate(
            individual_synthetic_data, individual_reference_data
        )

        return TableEvaluation(
            individual_sqs=individual_score, ancestral_sqs=model_score
        )


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
