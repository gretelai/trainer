from collections import defaultdict
from typing import Any, Dict, List, Tuple

import category_encoders as ce
import faiss
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from sklearn.preprocessing import StandardScaler

from gretel_trainer.relational.core import MultiTableException, RelationalData


class AncestralStrategy:
    def __init__(self, model_type: str = "Amplify"):
        if model_type not in ("ACTGAN", "Amplify"):
            raise MultiTableException(
                f"Unsupported model type: {model_type}. Supported model types are `ACTGAN` and `Amplify`."
            )
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

        If the table does have parents, jobs depend on model type.
        For ACTGAN, returns a list of 20 jobs, each to generate 1 million records.
        For Amplify, builds a seed dataframe to use in generation.
        """
        if len(rel_data.get_parents(table)) == 0:
            requested_synth_count = (
                len(rel_data.get_table_data(table)) * record_size_ratio
            )
            return [{"num_records": requested_synth_count}]
        elif self._model_type == "ACTGAN":
            return [{"num_records": 1_000_000} for i in range(20)]
        else:
            seed_df = rel_data.build_seed_data_for_table(table, output_tables)
            return [{"seed_df": seed_df}]

    def collect_generation_results(
        self, results: List[pd.DataFrame], table_name: str, rel_data: RelationalData
    ) -> pd.DataFrame:
        """
        For ACTGAN: TODO.

        For Amplify, concatenates all results, which should just be a list of one element.
        """
        if self._model_type == "ACTGAN":
            return _apply_faiss(results, table_name, rel_data)
        else:
            return pd.concat(results)


def _apply_faiss(
    results: List[pd.DataFrame], table_name: str, rel_data: RelationalData
) -> pd.DataFrame:
    source_seed = rel_data.build_seed_data_for_table(table_name)
    if source_seed is None:
        raise MultiTableException(f"Could not build source seed for table {table_name}")

    all_multigenerational_synthetic_data = pd.concat(results)
    synthetic_seed = _distill_to_seed(
        all_multigenerational_synthetic_data, rel_data, source_seed
    )

    synth_faiss, source_faiss = _prep_for_faiss(synthetic_seed, source_seed)

    index = faiss.IndexFlatL2(len(source_seed.columns))
    index.add(synth_faiss)
    number_of_nearest_neighbors = 1
    D, I = index.search(source_faiss, number_of_nearest_neighbors)

    dist_list = []
    results_list = []
    for i in range(len(D)):
        dist_list.append(D[i][0])
        results_list.append(I[i][0])

    results_stats = defaultdict(lambda: 1)
    for value in results_list:
        results_stats[value] += 1

    # Gather the count 1, count 2, etc lists
    # This is a little wacky, can't think of a more efficient way to gather indices occuring multiple times
    count_lists = defaultdict(list)
    for rsk, rsv in results_stats.items():
        if rsv > 1:
            for j in range(rsv, 1, -1):
                count_lists[j].append(rsk)

    # Gather the matches that occur at least once
    x = all_multigenerational_synthetic_data.index.isin(results_list)
    matches = all_multigenerational_synthetic_data.loc[x]

    # Gather the matches that occur more than once
    for clv in count_lists.values():
        x = all_multigenerational_synthetic_data.index.isin(clv)
        synth_df_next = all_multigenerational_synthetic_data.loc[x]
        matches = pd.concat([matches, synth_df_next])

    # Now reset the foreign keys to match the primary keys
    for afk_map in rel_data.get_ancestral_foreign_key_maps(table_name):
        fk, referenced = afk_map
        matches[fk] = matches[referenced]

    return matches


# TODO: can we just do one or the other of these filters? Do we need both?
def _distill_to_seed(
    df: pd.DataFrame, rel_data: RelationalData, source_seed: pd.DataFrame
) -> pd.DataFrame:
    """
    Given a multigenerational dataframe, returns a copy with only non-primary-key ancestral columns retained.
    """
    columns_to_retain = [
        column
        for column in df.columns
        if rel_data.is_ancestral_column(column)
        and not rel_data.is_ancestral_primary_key(column)
    ]
    return df.filter(columns_to_retain).filter(source_seed.columns)


def _prep_for_faiss(
    syn: pd.DataFrame, src: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Concatenates synthetic and source dataframes. Encodes and standardizes all values.
    Splits result back into encoded-and-standardized synthetic and source dataframes.
    Formats both output dataframes for faiss.
    """
    df = pd.concat(syn, src)

    nominal_columns = list(df.select_dtypes(include=["object", "category"]).columns)
    if len(nominal_columns) > 0:
        df_cat = df.reindex(columns=nominal_columns).fillna("Missing")
        df_cat_labels = pd.DataFrame(ce.BinaryEncoder().fit_transform(df_cat))
    else:
        df_cat_labels = pd.DataFrame()

    numeric_columns = [col for col in df.columns if col not in nominal_columns]
    if len(numeric_columns) > 0:
        df_num = df.reindex(columns=numeric_columns)
        df_num = df_num.fillna(df_num.median())
    else:
        df_num = pd.DataFrame()

    new_df = pd.concat([df_cat_labels, df_num], axis=1, sort=False)
    new_df = pd.DataFrame(StandardScaler().fit_transform(new_df))

    def _format_for_faiss(df: pd.DataFrame) -> pd.DataFrame:
        return np.ascontiguousarray(df.to_numpy().astype("float32"))

    return (
        _format_for_faiss(new_df.head(len(syn))),
        _format_for_faiss(new_df.tail(len(src))),
    )


def _is_highly_unique_categorical(col: str, df: pd.DataFrame) -> bool:
    return is_string_dtype(df[col]) and _percent_unique(col, df) >= 0.7


def _percent_unique(col: str, df: pd.DataFrame) -> float:
    col_no_nan = df[col].dropna()
    total = len(col_no_nan)
    distinct = col_no_nan.nunique()

    return distinct / total
