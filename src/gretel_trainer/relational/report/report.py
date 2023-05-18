from __future__ import annotations

import datetime
from dataclasses import dataclass
from functools import cached_property
from math import ceil
from pathlib import Path
from typing import Optional

import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader

from gretel_trainer.relational.core import ForeignKey, RelationalData, Scope
from gretel_trainer.relational.report.figures import (
    PRIVACY_LEVEL_VALUES,
    gauge_and_needle_chart,
)
from gretel_trainer.relational.table_evaluation import TableEvaluation

_TEMPLATE_DIR = str(Path(__file__).parent)
_TEMPLATE_FILE = "report_template.html"


class ReportRenderer:
    def __init__(self):
        file_loader = FileSystemLoader(_TEMPLATE_DIR)
        env = Environment(loader=file_loader)
        self.template = env.get_template(_TEMPLATE_FILE)

    def render(self, presenter: ReportPresenter) -> str:
        return self.template.render(presenter=presenter)


@dataclass
class ReportTableData:
    table: str
    pk: list[str]
    fks: list[ForeignKey]


@dataclass
class ReportPresenter:
    rel_data: RelationalData
    now: datetime.datetime
    run_identifier: str
    evaluations: dict[str, TableEvaluation]

    @property
    def generated_at(self) -> str:
        return self.now.strftime("%Y-%m-%d")

    @property
    def copyright_year(self) -> str:
        return self.now.strftime("%Y")

    @cached_property
    def composite_sqs_score_and_grade(self) -> tuple[Optional[int], str]:
        # Add up all the non-None SQS scores and track how many there are.
        _total_score = 0
        _num_scores = 0
        for eval in self.evaluations.values():
            if eval.individual_sqs is not None:
                _total_score += eval.individual_sqs
                _num_scores += 1
            if eval.cross_table_sqs is not None:
                _total_score += eval.cross_table_sqs
                _num_scores += 1

        # Take the average.
        if _num_scores > 0:
            score = ceil(_total_score / _num_scores)
        # Or tell the user the bad news.
        else:
            score = None
        return (score, self.sqs_score_to_grade(score))

    @property
    def composite_sqs_label(self) -> str:
        _formatted_grade = self.css_label_format(self.composite_sqs_score_and_grade[1])
        return f"label__{_formatted_grade}"

    @property
    def composite_sqs_figure(self) -> go.Figure:
        score, grade = self.composite_sqs_score_and_grade
        return gauge_and_needle_chart(score)

    @cached_property
    def composite_ppl_score_and_grade(self) -> tuple[Optional[int], str]:
        # Collect all the non-None PPLs, individual and cross-table.
        scores = [
            eval.individual_ppl
            for eval in self.evaluations.values()
            if eval.individual_ppl is not None
        ]
        scores += [
            eval.cross_table_ppl
            for eval in self.evaluations.values()
            if eval.cross_table_ppl is not None
        ]
        # Take the min, our "weakest link"
        if len(scores) > 0:
            score = min(scores)
            GRADES = ["Normal", "Good", "Very Good", "Excellent"]
            if 0 <= score < 0.5:
                return (score, GRADES[0])
            if 0.5 <= score < 2.5:
                return (score, GRADES[1])
            if 2.5 <= score < 4.5:
                return (score, GRADES[2])
            return (score, GRADES[3])
        # Or tell the user the bad news.
        else:
            GRADE_UNAVAILABLE = "Unavailable"
            return (None, GRADE_UNAVAILABLE)

    @property
    def composite_ppl_label(self) -> str:
        _formatted_grade = self.css_label_format(self.composite_ppl_score_and_grade[1])
        return f"label__privacy__{_formatted_grade}"

    @property
    def composite_ppl_figure(self) -> go.Figure:
        score, grade = self.composite_ppl_score_and_grade
        ppl_score_map = {0: 30, 1: 46, 2: 54, 3: 66, 4: 74, 5: 86, 6: 94}
        if score is None:
            ppl_score = 30
        else:
            ppl_score = ppl_score_map.get(score, 30)
        return gauge_and_needle_chart(
            ppl_score,
            display_score=False,
            marker_colors=[s["color"] for s in PRIVACY_LEVEL_VALUES],
        )

    @property
    def report_table_data(self) -> list[ReportTableData]:
        table_data = []
        for table in self.rel_data.list_all_tables(Scope.PUBLIC):
            pk = self.rel_data.get_primary_key(table)
            fks = self.rel_data.get_foreign_keys(table, rename_invented_tables=True)
            table_data.append(ReportTableData(table=table, pk=pk, fks=fks))

        # Sort tables alphabetically because that's nice.
        table_data = sorted(table_data, key=lambda x: x.table)
        return table_data

    # Helper, making it "just a method" so it is easily accessible in jinja template.
    def sqs_score_to_grade(self, score: Optional[int]) -> str:
        GRADES = ["Very Poor", "Poor", "Moderate", "Good", "Excellent"]
        GRADE_UNAVAILABLE = "Unavailable"
        if score is None:
            return GRADE_UNAVAILABLE
        idx = score // 20
        # Constrain to [0,4]
        idx = max(0, min(4, idx))
        return GRADES[idx]

    def css_label_format(self, grade: str) -> str:
        return grade.lower().replace(" ", "-")
