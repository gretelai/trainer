from typing import Any, Dict, List

import pandas as pd
from pandas.api.types import is_string_dtype

from gretel_trainer.relational.core import RelationalData


class AncestralStrategy:
    def __init__(self, model_type: str = "Amplify"):
        self._model_type = model_type

    def prepare_training_data(
        self, table_name: str, rel_data: RelationalData
    ) -> pd.DataFrame:
        """
        Returns table data with all ancestors added, minus
        all primary and foreign keys and any highly-unique
        categorical fields from parents.
        """
        data = rel_data.get_table_data_with_ancestors(table_name)
        columns_to_drop = []

        columns_to_drop.extend(rel_data.list_multigenerational_keys(table_name))
        for column in data.columns:
            if rel_data.is_ancestral_column(column) and _is_highly_unique_categorical(
                column, data
            ):
                columns_to_drop.append(column)

        return data.drop(columns=columns_to_drop)

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
        if self._model_type == "ACTGAN":
            return [{"num_records": 1_000_000} for i in range(20)]
        elif len(rel_data.get_parents(table)) == 0:
            requested_synth_count = len(rel_data.get_table_data(table)) * record_size_ratio
            return [{"num_records": requested_synth_count}]
        else:
            seed_df = rel_data.build_seed_data_for_table(table, output_tables)
            return [{"seed_df": seed_df}]

    def collect_generation_results(
        self, results: List[pd.DataFrame], table_name: str, rel_data: RelationalData
    ) -> pd.DataFrame:
        if self._model_type == "ACTGAN":
            return _trim(results, table_name, rel_data)
        else:
            return pd.concat(results)


def _trim(
    results: List[pd.DataFrame], table_name: str, rel_data: RelationalData
) -> pd.DataFrame:
    # TODO
    return pd.concat(results)


def _is_highly_unique_categorical(col: str, df: pd.DataFrame) -> bool:
    return is_string_dtype(df[col]) and _percent_unique(col, df) >= 0.7


def _percent_unique(col: str, df: pd.DataFrame) -> float:
    col_no_nan = df[col].dropna()
    total = len(col_no_nan)
    distinct = col_no_nan.nunique()

    return distinct / total
