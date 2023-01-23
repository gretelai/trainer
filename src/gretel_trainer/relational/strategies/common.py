import json
import logging
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import pandas as pd
import smart_open
from gretel_client.evaluation.quality_report import QualityReport
from gretel_client.projects.models import Model
from sklearn import preprocessing

from gretel_trainer.relational.core import RelationalData

logger = logging.getLogger(__name__)


def get_quality_report(
    source_data: pd.DataFrame, synth_data: pd.DataFrame
) -> QualityReport:
    report = QualityReport(data_source=synth_data, ref_data=source_data)
    report.run()
    return report


def write_report(report: QualityReport, table_name: str, working_dir: Path) -> None:
    html_path = working_dir / f"expanded_evaluation_{table_name}.html"
    json_path = working_dir / f"expanded_evaluation_{table_name}.json"

    with open(html_path, "w") as f:
        f.write(report.as_html)
    with open(json_path, "w") as f:
        f.write(json.dumps(report.as_dict))


def download_artifacts(
    model: Model, table_name: str, working_dir: Path
) -> Optional[Path]:
    """
    Downloads all model artifacts to a subdirectory in the working directory.
    Returns the artifact directory path when successful.
    """
    target_dir = working_dir / f"artifacts_{table_name}"
    logger.info(f"Downloading model artifacts for {table_name}")
    try:
        model.download_artifacts(target_dir)
        return target_dir
    except:
        logger.warning(f"Failed to download model artifacts for {table_name}")
        return None


def read_report_json_data(
    model: Model, artifacts_dir: Optional[Path]
) -> Optional[Dict]:
    if artifacts_dir is not None:
        report_json_path = artifacts_dir / "report_json.json.gz"
        return json.loads(smart_open.open(report_json_path).read())
    else:
        return _get_report_json(model)


def _get_report_json(model: Model) -> Optional[Dict]:
    try:
        return json.loads(
            smart_open.open(model.get_artifact_link("report_json")).read()
        )
    except:
        logger.warning("Failed to fetch model evaluation report JSON.")
        return None


def get_sqs_score(model: Model) -> Optional[int]:
    summary = model.get_report_summary()
    if summary is None or summary.get("summary") is None:
        logger.warning("Failed to fetch model evaluation report summary.")
        return None

    sqs_score = None
    for stat in summary["summary"]:
        if stat["field"] == "synthetic_data_quality_score":
            sqs_score = stat["value"]

    return sqs_score


def label_encode_keys(
    rel_data: RelationalData, tables: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """
    Crawls tables for all key columns (primary and foreign). For each PK (and FK columns referencing it),
    runs all values through a LabelEncoder and updates tables' columns to use LE-transformed values.
    """
    for table_name, df in tables.items():
        primary_key = rel_data.get_primary_key(table_name)
        if primary_key is not None:
            # Get a set of the tables and columns in `tables` referencing this PK
            fk_references: Set[Tuple[str, str]] = set()
            for descendant in rel_data.get_descendants(table_name):
                desc_table = tables.get(descendant)
                if desc_table is None:
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

            for col_value in df[primary_key]:
                source_values.add(col_value)

            for fk_ref in fk_references:
                fk_tbl, fk_col = fk_ref
                for col_value in tables[fk_tbl][fk_col]:
                    source_values.add(col_value)

            # Fit a label encoder on all values
            le = preprocessing.LabelEncoder()
            le.fit(list(source_values))

            # Update PK and FK columns using the label encoder
            df[primary_key] = le.transform(df[primary_key])

            for fk_ref in fk_references:
                fk_tbl, fk_col = fk_ref
                tables[fk_tbl][fk_col] = le.transform(tables[fk_tbl][fk_col])

    return tables
