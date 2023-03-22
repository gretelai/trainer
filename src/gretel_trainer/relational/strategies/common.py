import json
import logging
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import pandas as pd
import smart_open
from gretel_client.projects.models import Model
from sklearn import preprocessing

from gretel_trainer.relational.core import RelationalData
from gretel_trainer.relational.sdk_extras import download_file_artifact

logger = logging.getLogger(__name__)


def download_artifacts(model: Model, out_filepath: Path, table_name: str) -> None:
    """
    Downloads all model artifacts to a subdirectory in the working directory.
    Returns the artifact directory path when successful.
    """
    legend = {"html": "report", "json": "report_json"}

    for filetype, artifact_name in legend.items():
        out_path = f"{out_filepath}.{filetype}"
        download_file_artifact(model, artifact_name, out_path)


def read_report_json_data(model: Model, report_path: Path) -> Optional[Dict]:
    full_path = f"{report_path}.json"
    try:
        return json.loads(smart_open.open(full_path).read())
    except:
        return _get_report_json(model)


def _get_report_json(model: Model) -> Optional[Dict]:
    try:
        return json.loads(
            smart_open.open(model.get_artifact_link("report_json")).read()
        )
    except:
        logger.warning("Failed to fetch model evaluation report JSON.")
        return None


def label_encode_keys(
    rel_data: RelationalData, tables: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """
    Crawls tables for all key columns (primary and foreign). For each PK (and FK columns referencing it),
    runs all values through a LabelEncoder and updates tables' columns to use LE-transformed values.
    """
    for table_name, df in tables.items():
        primary_key = rel_data.get_primary_key(table_name)
        if primary_key is None:
            continue

        # Get a set of the tables and columns in `tables` referencing this PK
        fk_references: Set[Tuple[str, str]] = set()
        for descendant in rel_data.get_descendants(table_name):
            if tables.get(descendant) is None:
                continue
            fks = rel_data.get_foreign_keys(descendant)
            for fk in fks:
                if (
                    fk.parent_table_name == table_name
                    and fk.parent_column_name == primary_key
                ):
                    fk_references.add((descendant, fk.column_name))

        # Collect column values from PK and FK columns into a set
        source_values = set()
        source_values.update(df[primary_key].to_list())
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
        df[primary_key] = le.transform(df[primary_key])

        for fk_ref in fk_references:
            fk_tbl, fk_col = fk_ref
            fk_df = tables.get(fk_tbl)
            if fk_df is None:
                continue
            fk_df[fk_col] = le.transform(fk_df[fk_col])

    return tables
