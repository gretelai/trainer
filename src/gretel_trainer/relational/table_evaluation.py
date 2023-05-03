import json
from dataclasses import dataclass, field
from typing import Literal, Optional, Union, overload

_SQS = "synthetic_data_quality_score"
_PPL = "privacy_protection_level"
_SCORE = "score"
_GRADE = "grade"


@dataclass
class TableEvaluation:
    cross_table_report_json: Optional[dict] = field(default=None, repr=False)
    individual_report_json: Optional[dict] = field(default=None, repr=False)

    def is_complete(self) -> bool:
        return (
            self.cross_table_report_json is not None
            and self.cross_table_sqs is not None
            and self.individual_report_json is not None
            and self.individual_sqs is not None
        )

    @overload
    def _field_from_json(
        self, report_json: Optional[dict], entry: str, field: Literal["score"]
    ) -> Optional[int]:
        ...

    @overload
    def _field_from_json(
        self, report_json: Optional[dict], entry: str, field: Literal["grade"]
    ) -> Optional[str]:
        ...

    def _field_from_json(
        self, report_json: Optional[dict], entry: str, field: str
    ) -> Optional[Union[int, str]]:
        if report_json is None:
            return None
        else:
            return report_json.get(entry, {}).get(field)

    @property
    def cross_table_sqs(self) -> Optional[int]:
        return self._field_from_json(self.cross_table_report_json, _SQS, _SCORE)

    @property
    def cross_table_sqs_grade(self) -> Optional[str]:
        return self._field_from_json(self.cross_table_report_json, _SQS, _GRADE)

    @property
    def cross_table_ppl(self) -> Optional[int]:
        return self._field_from_json(self.cross_table_report_json, _PPL, _SCORE)

    @property
    def cross_table_ppl_grade(self) -> Optional[str]:
        return self._field_from_json(self.cross_table_report_json, _PPL, _GRADE)

    @property
    def individual_sqs(self) -> Optional[int]:
        return self._field_from_json(self.individual_report_json, _SQS, _SCORE)

    @property
    def individual_sqs_grade(self) -> Optional[str]:
        return self._field_from_json(self.individual_report_json, _SQS, _GRADE)

    @property
    def individual_ppl(self) -> Optional[int]:
        return self._field_from_json(self.individual_report_json, _PPL, _SCORE)

    @property
    def individual_ppl_grade(self) -> Optional[str]:
        return self._field_from_json(self.individual_report_json, _PPL, _GRADE)

    def __repr__(self) -> str:
        d = {}
        if self.cross_table_report_json is not None:
            d["cross_table"] = {
                "sqs": {
                    "score": self.cross_table_sqs,
                    "grade": self.cross_table_sqs_grade,
                },
                "ppl": {
                    "score": self.cross_table_ppl,
                    "grade": self.cross_table_ppl_grade,
                },
            }
        if self.individual_report_json is not None:
            d["individual"] = {
                "sqs": {
                    "score": self.individual_sqs,
                    "grade": self.individual_sqs_grade,
                },
                "ppl": {
                    "score": self.individual_ppl,
                    "grade": self.individual_ppl_grade,
                },
            }
        return json.dumps(d)
