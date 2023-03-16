from __future__ import annotations

import datetime
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
from backports.cached_property import cached_property
from jinja2 import Environment, FileSystemLoader

from gretel_trainer.relational.core import ForeignKey, RelationalData, TableEvaluation
from gretel_trainer.relational.report.figures import (
    PRIVACY_LEVEL_VALUES,
    gauge_and_needle_chart,
)

_TEMPLATE_DIR = str(Path(__file__).parent)
# _TEMPLATE_FILE = "report_template.html"
_TEMPLATE_FILE = "report_template_two.html"


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
    pk: Optional[str]
    fks: List[ForeignKey]


@dataclass
class ReportPresenter:
    rel_data: RelationalData
    now: datetime.datetime
    run_identifier: str
    evaluations: Dict[str, TableEvaluation]

    @property
    def generated_at(self) -> str:
        return self.now.strftime("%Y-%m-%d")

    @property
    def copyright_year(self) -> str:
        return self.now.strftime("%Y")

    # @property
    # def relationships(self) -> Relationships:
    #     return _table_relationships(self.rel_data)

    @cached_property
    def composite_sqs_score_and_grade(self) -> Tuple[Optional[int], str]:
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
            score = int(_total_score / _num_scores)
        # Or tell the user the bad news.
        else:
            score = None
        return (score, self.sqs_score_to_grade(score))

    @property
    def composite_sqs_figure(self) -> go.Figure:
        score, grade = self.composite_sqs_score_and_grade
        return gauge_and_needle_chart(score)

    @cached_property
    def composite_ppl_score_and_grade(self) -> Tuple[Optional[int], str]:
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
    def composite_ppl_figure(self) -> go.Figure:
        score, grade = self.composite_ppl_score_and_grade
        return gauge_and_needle_chart(
            score,
            display_score=False,
            marker_colors=[s["color"] for s in PRIVACY_LEVEL_VALUES],
        )

    @property
    def report_table_data(self) -> List[ReportTableData]:
        table_data = []
        for table in self.rel_data.list_all_tables():
            pk = self.rel_data.get_primary_key(table)
            fks = self.rel_data.get_foreign_keys(table)
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


# @dataclass
# class Cell:
#     """
#     Dataclass to hold the formatting properties of the cell for rendering Table Relationships table

#     Attributes:
#         value (str): The text value of the cell
#         background_color (str, optional): Optional background color, in hexcode format
#         font_weight (str, optional): Optional modifier to font
#     """

#     value: str
#     background_color: Optional[str] = None
#     font_weight: Optional[str] = None

#     @property
#     def style(self) -> str:
#         styles = []
#         if self.background_color is not None:
#             styles.append(f"background-color: {self.background_color}")
#         if self.font_weight is not None:
#             styles.append(f"font-weight: {self.font_weight}")
#         return "; ".join(styles)


# @dataclass
# class Relationships:
#     columns: List[str]
#     data: List[Cell]


# def _table_relationships(rel_data: RelationalData) -> Relationships:
#     pk_colors = _assign_primary_key_colors(rel_data)

#     df_data = []
#     tables = []

#     for table in rel_data.list_all_tables():
#         tables.append(table)

#         seen = set()
#         fields = []

#         # Add primary key
#         pk = rel_data.get_primary_key(table)
#         if pk is None:
#             fields.append(
#                 Cell(
#                     value="(no primary key)",
#                 )
#             )
#         else:
#             seen.add(pk)
#             fields.append(
#                 Cell(
#                     value=pk,
#                     background_color=pk_colors[table],
#                     font_weight="bold",
#                 )
#             )

#         # Add foreign keys
#         fks = rel_data.get_foreign_keys(table)
#         for fk in fks:
#             seen.add(fk.column_name)
#             fields.append(
#                 Cell(
#                     value=fk.column_name,
#                     background_color=pk_colors[fk.parent_table_name],
#                 )
#             )

#         # Add other fields
#         for column in rel_data.get_table_data(table).columns:
#             if column not in seen:
#                 fields.append(
#                     Cell(
#                         value=column,
#                     )
#                 )

#         df_data.append(fields)

#     df = pd.DataFrame(df_data, index=tables).transpose().fillna("")
#     split = df.to_dict("split")

#     return Relationships(
#         columns=split["columns"],
#         data=split["data"],
#     )


# def _assign_primary_key_colors(rel_data: RelationalData) -> Dict[str, str]:
#     colors = itertools.cycle(
#         [
#             "#F48FB1",
#             "#C5E1A5",
#             "#FFCC80",
#             "#CE93D8",
#             "#FFF59D",
#             "#90CAF9",
#             "#FFE082",
#             "#B39DDB",
#             "#A5D6A7",
#             "#80DEEA",
#         ]
#     )
#     color_dict = {}
#     for table in rel_data.list_all_tables():
#         if rel_data.get_primary_key(table) is not None:
#             color_dict[table] = next(colors)
#     return color_dict
