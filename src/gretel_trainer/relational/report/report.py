from __future__ import annotations

import datetime
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from gretel_trainer.relational.core import RelationalData, TableEvaluation
from gretel_trainer.relational.multi_table import MultiTable

_TEMPLATE_DIR = str(Path(__file__).parent)


def create_report(multitable: MultiTable) -> None:
    presenter = ReportPresenter(
        rel_data=multitable.relational_data,
        evaluations=multitable.evaluations,
        now=datetime.datetime.utcnow(),
    )
    output_path = multitable._working_dir / "multitable_report.html"
    with open(output_path, "w") as report:
        html_content = ReportRenderer().render(presenter)
        report.write(html_content)


class ReportRenderer:
    def __init__(self):
        file_loader = FileSystemLoader(_TEMPLATE_DIR)
        env = Environment(loader=file_loader)
        self.template = env.get_template("report_template.html")

    def render(self, presenter: ReportPresenter) -> str:
        return self.template.render(presenter=presenter)


@dataclass
class ReportPresenter:
    rel_data: RelationalData
    now: datetime.datetime
    evaluations: Dict[str, TableEvaluation]

    @property
    def generated_at(self) -> str:
        return self.now.strftime("%m/%d/%Y, %H:%M")

    @property
    def copyright_year(self) -> str:
        return self.now.strftime("%Y")

    @property
    def relationships(self) -> Relationships:
        return _table_relationships(self.rel_data)


@dataclass
class Cell:
    """
    Dataclass to hold the formatting properties of the cell for rendering Table Relationships table

    Attributes:
        value (str): The text value of the cell
        background_color (str, optional): Optional background color, in hexcode format
        font_weight (str, optional): Optional modifier to font
    """

    value: str
    background_color: Optional[str] = None
    font_weight: Optional[str] = None

    @property
    def style(self) -> str:
        styles = []
        if self.background_color is not None:
            styles.append(f"background-color: {self.background_color}")
        if self.font_weight is not None:
            styles.append(f"font-weight: {self.font_weight}")
        return "; ".join(styles)


@dataclass
class Relationships:
    columns: List[str]
    data: List[Cell]


def _table_relationships(rel_data: RelationalData) -> Relationships:
    pk_colors = _assign_primary_key_colors(rel_data)

    df_data = []
    tables = []

    for table in rel_data.list_all_tables():
        tables.append(table)

        seen = set()
        fields = []

        # Add primary key
        pk = rel_data.get_primary_key(table)
        if pk is None:
            fields.append(
                Cell(
                    value="(no primary key)",
                )
            )
        else:
            seen.add(pk)
            fields.append(
                Cell(
                    value=pk,
                    background_color=pk_colors[table],
                    font_weight="bold",
                )
            )

        # Add foreign keys
        fks = rel_data.get_foreign_keys(table)
        for fk in fks:
            seen.add(fk.column_name)
            fields.append(
                Cell(
                    value=fk.column_name,
                    background_color=pk_colors[fk.parent_table_name],
                )
            )

        # Add other fields
        for column in rel_data.get_table_data(table).columns:
            if column not in seen:
                fields.append(
                    Cell(
                        value=column,
                    )
                )

        df_data.append(fields)

    df = pd.DataFrame(df_data, index=tables).transpose().fillna("")
    split = df.to_dict("split")

    return Relationships(
        columns=split["columns"],
        data=split["data"],
    )


def _assign_primary_key_colors(rel_data: RelationalData) -> Dict[str, str]:
    colors = itertools.cycle(
        ["#ABEBC6", "#ABC6EB", "#EBABC6", "#EBC6AB", "#C6ABEB", "#C6EBAB"]
    )
    color_dict = {}
    for table in rel_data.list_all_tables():
        if rel_data.get_primary_key(table) is not None:
            color_dict[table] = next(colors)
    return color_dict
