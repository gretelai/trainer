import pandas as pd
from gretel_client.evaluation.quality_report import QualityReport


def get_sqs_via_evaluate(data_source: pd.DataFrame, ref_data: pd.DataFrame) -> int:
    report = QualityReport(data_source=data_source, ref_data=ref_data)
    report.run()
    return report.peek()["score"]
