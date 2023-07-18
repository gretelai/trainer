import json
import logging
import random
from pathlib import Path
from typing import Optional

import pandas as pd
import smart_open
from gretel_client.projects.models import Model
from sklearn import preprocessing

from gretel_trainer.relational.core import MultiTableException, RelationalData
from gretel_trainer.relational.sdk_extras import ExtendedGretelSDK

logger = logging.getLogger(__name__)


def download_artifacts(
    model: Model, out_filepath: Path, extended_sdk: ExtendedGretelSDK
) -> None:
    """
    Downloads all model artifacts to a subdirectory in the working directory.
    """
    legend = {"html": "report", "json": "report_json"}

    for filetype, artifact_name in legend.items():
        out_path = f"{out_filepath}.{filetype}"
        extended_sdk.download_file_artifact(model, artifact_name, out_path)


def read_report_json_data(model: Model, report_path: Path) -> Optional[dict]:
    full_path = f"{report_path}.json"
    try:
        return json.loads(smart_open.open(full_path).read())
    except:
        return _get_report_json(model)


def _get_report_json(model: Model) -> Optional[dict]:
    try:
        return json.loads(
            smart_open.open(model.get_artifact_link("report_json")).read()
        )
    except:
        logger.warning("Failed to fetch model evaluation report JSON.")
        return None


def label_encode_keys(
    rel_data: RelationalData,
    tables: dict[str, pd.DataFrame],
    omit: Optional[list[str]] = None,
) -> dict[str, pd.DataFrame]:
    """
    Crawls tables for all key columns (primary and foreign). For each PK (and FK columns referencing it),
    runs all values through a LabelEncoder and updates tables' columns to use LE-transformed values.
    """
    omit = omit or []
    for table_name in rel_data.list_tables_parents_before_children():
        if table_name in omit:
            continue

        df = tables.get(table_name)
        if df is None:
            continue

        for primary_key_column in rel_data.get_primary_key(table_name):
            # Get a set of the tables and columns in `tables` referencing this PK
            fk_references: set[tuple[str, str]] = set()
            for descendant in rel_data.get_descendants(table_name):
                if tables.get(descendant) is None:
                    continue
                fks = rel_data.get_foreign_keys(descendant)
                for fk in fks:
                    if fk.parent_table_name != table_name:
                        continue

                    for i in range(len(fk.columns)):
                        if fk.parent_columns[i] == primary_key_column:
                            fk_references.add((descendant, fk.columns[i]))

            # Collect column values from PK and FK columns into a set
            source_values = set()
            source_values.update(df[primary_key_column].to_list())
            for fk_ref in fk_references:
                fk_tbl, fk_col = fk_ref
                fk_df = tables.get(fk_tbl)
                if fk_df is None:
                    continue
                source_values.update(fk_df[fk_col].to_list())

            # Fit a label encoder on all values
            le = preprocessing.LabelEncoder()
            le.fit(list(source_values))

            # Update PK and FK columns using the label encoder
            df[primary_key_column] = le.transform(df[primary_key_column])

            for fk_ref in fk_references:
                fk_tbl, fk_col = fk_ref
                fk_df = tables.get(fk_tbl)
                if fk_df is None:
                    continue
                fk_df[fk_col] = le.transform(fk_df[fk_col])

    return tables


def make_composite_pks(
    table_name: str,
    rel_data: RelationalData,
    primary_key: list[str],
    synth_row_count: int,
) -> list[dict]:
    # Given the randomness involved in this process, it is possible for this function to generate
    # fewer unique composite keys than desired to completely fill the dataframe (i.e. the length
    # of the tuple values in the dictionary may be < synth_row_count). It is the client's
    # responsibility to check for this and drop synthetic records if necessary to fit.
    table_data = rel_data.get_table_data(table_name)
    original_primary_key = rel_data.get_primary_key(table_name)

    pk_component_frequencies = {
        col: get_frequencies(table_data, [col]) for col in original_primary_key
    }

    # each key in new_cols is a column name, and each value is a list of
    # column values. The values are a contiguous list of integers, with
    # each integer value appearing 1-N times to match the frequencies of
    # (original source) values' appearances in the source data.
    new_cols: dict[str, list] = {}
    for i, col in enumerate(primary_key):
        freqs = pk_component_frequencies[original_primary_key[i]]
        next_freq = 0
        next_key = 0
        new_col_values = []

        while len(new_col_values) < synth_row_count:
            for i in range(freqs[next_freq]):
                new_col_values.append(next_key)
            next_key += 1
            next_freq += 1
            if next_freq == len(freqs):
                next_freq = 0

        # A large frequency may have added more values than we need,
        # so trim to synth_row_count
        new_cols[col] = new_col_values[0:synth_row_count]

    # Shuffle for realism
    for col_name, col_values in new_cols.items():
        random.shuffle(col_values)

    # Zip the individual columns into a list of records.
    # Each element in the list is a composite key dict.
    composite_keys: list[dict] = []
    for i in range(synth_row_count):
        comp_key = {}
        for col_name, col_values in new_cols.items():
            comp_key[col_name] = col_values[i]
        composite_keys.append(comp_key)

    # The zip above may not have produced unique composite key dicts.
    # Using the most unique column (to give us the most options), try
    # changing a value to "resolve" candidate composite keys to unique combinations.
    cant_resolve = 0
    seen: set[str] = set()
    final_synthetic_composite_keys: list[dict] = []
    most_unique_column = _get_most_unique_column(primary_key, pk_component_frequencies)

    for i in range(synth_row_count):
        y = i + 1
        if y == len(composite_keys):
            y = 0

        comp_key = composite_keys[i]

        while str(comp_key) in seen and y != i:
            last_val = new_cols[most_unique_column][y]
            y += 1
            if y == len(composite_keys):
                y = 0
            comp_key[most_unique_column] = last_val
        if str(comp_key) in seen:
            cant_resolve += 1
        else:
            final_synthetic_composite_keys.append(comp_key)
            seen.add(str(comp_key))

    return final_synthetic_composite_keys


def _get_most_unique_column(pk: list[str], col_freqs: dict[str, list]) -> str:
    most_unique = None
    max_length = 0
    for col, freqs in col_freqs.items():
        if len(freqs) > max_length:
            most_unique = col

    if most_unique is None:
        raise MultiTableException(
            f"Failed to identify most unique column from column frequencies: {col_freqs}"
        )

    # The keys in col_freqs are always the source column names from the original primary key.
    # Meanwhile, `pk` could be either the same (independent strategy) or in multigenerational
    # format (ancestral strategy). We need to return the column name in the format matching
    # the rest of the synthetic data undergoing post-processing.
    idx = list(col_freqs.keys()).index(most_unique)
    return pk[idx]


def get_frequencies(table_data: pd.DataFrame, cols: list[str]) -> list[int]:
    return list(table_data.groupby(cols).size().reset_index()[0])
