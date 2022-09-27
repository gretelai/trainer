from dataclasses import dataclass

import pandas as pd

from gretel_client.evaluation.quality_report import QualityReport


@dataclass
class GretelEvaluate:
    def get_sqs_score(self, synthetic: pd.DataFrame, reference: pd.DataFrame) -> int:
        report = QualityReport(data_source=synthetic, ref_data=reference)
        report.run()
        return report.peek()["score"]
