import json
import logging
import math
import random
from pathlib import Path
from typing import Optional

import pandas as pd
import smart_open
from gretel_client.projects.models import Model
from sklearn import preprocessing

from gretel_trainer.relational.core import RelationalData
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
    rel_data: RelationalData, tables: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """
    Crawls tables for all key columns (primary and foreign). For each PK (and FK columns referencing it),
    runs all values through a LabelEncoder and updates tables' columns to use LE-transformed values.
    """
    for table_name in rel_data.list_tables_parents_before_children():
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


def make_composite_pk_columns(
    table_name: str,
    rel_data: RelationalData,
    primary_key: list[str],
    synth_row_count: int,
    record_size_ratio: float,
) -> list[tuple]:
    source_pk_columns = rel_data.get_table_data(table_name)[primary_key]
    unique_counts = source_pk_columns.nunique(axis=0)
    new_key_columns_values = []
    for col in primary_key:
        synth_values_count = math.ceil(unique_counts[col] * record_size_ratio)
        new_key_columns_values.append(range(synth_values_count))

    results = set()
    while len(results) < synth_row_count:
        key_combination = tuple(
            [random.choice(vals) for vals in new_key_columns_values]
        )
        results.add(key_combination)

    return list(zip(*results))


def get_frequencies(table_data: pd.DataFrame, cols: list[str]) -> list[int]:
    return list(table_data.groupby(cols).size().reset_index()[0])
