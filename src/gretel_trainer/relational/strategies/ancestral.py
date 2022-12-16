import pandas as pd

from pandas.api.types import is_string_dtype
from gretel_trainer.relational.core import RelationalData


class AncestralStrategy:
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
            if rel_data.is_ancestral_column(column) and _is_highly_unique_categorical(column, data):
                columns_to_drop.append(column)

        return data.drop(columns=columns_to_drop)


def _is_highly_unique_categorical(col: str, df: pd.DataFrame) -> bool:
    return is_string_dtype(df[col]) and _percent_unique(col, df) >= 0.7


def _percent_unique(col: str, df: pd.DataFrame) -> float:
    col_no_nan = df[col].dropna()
    total = len(col_no_nan)
    distinct = col_no_nan.unique()

    return distinct / total
