import json
import logging
from typing import Dict, Optional

import pandas as pd
import smart_open
from gretel_client.evaluation.quality_report import QualityReport
from gretel_client.projects.models import Model

logger = logging.getLogger(__name__)


def get_quality_report(
    source_data: pd.DataFrame, synth_data: pd.DataFrame
) -> QualityReport:
    report = QualityReport(data_source=synth_data, ref_data=source_data)
    report.run()
    return report


def get_report_json(model: Model) -> Optional[Dict]:
    try:
        return json.loads(
            smart_open.open(model.get_artifact_link("report_json")).read()
        )
    except:
        logger.warning("Failed to fetch model evaluation report JSON.")
        return None


def get_report_html(model: Model) -> Optional[str]:
    try:
        return smart_open.open(model.get_artifact_link("report")).read()
    except:
        logger.warning("Failed to fetch model evaluation report HTML.")
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
