import datetime
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
from jinja2 import Environment, FileSystemLoader, Template

from gretel_trainer.relational import MultiTable

logger = logging.getLogger(__name__)

_TEMPLATE_PATH = str(Path(__file__).parent)

file_loader = FileSystemLoader(_TEMPLATE_PATH)
env = Environment(loader=file_loader)
template = env.get_template("report_template.html")
template.globals["now"] = datetime.datetime.utcnow()


class Report:
    def __init__(self, multitable: MultiTable):
        self.multitable = multitable
        self.rel_data = multitable.relational_data
        self.eval_dict = multitable.evaluations
        self.fields_dict = self.create_fields_dict()
        self.proj_name = multitable._project.display_name
        self.tables = multitable.relational_data.list_all_tables()
        self.num_tables = multitable.relational_data.debug_summary["table_count"]
        self.num_fkeys = multitable.relational_data.debug_summary["foreign_key_count"]
        self.max_depth = multitable.relational_data.debug_summary["max_depth"]
        self.dir = multitable._working_dir

    def create_fields_dict(self):
        """
        Output:

        """
        columns_data = []
        tables = []

        # Start by assigning key colors
        colors_dict = assign_key_colors(self.rel_data)

        # For each table
        for table in self.rel_data:
            columns = []
            tables.append(table)

            # Add primary key, if exists
            pkey = self.rel_data.get_primary_key(table)
            if pkey is not None:
                # Format cell class for pkey and append - bolded, color from color_dict
                columns.append(Cell(value=pkey, color=colors_dict[table], bold="bold"))

            # Add any foreign keys
            fkeys = self.rel_data.get_foreign_keys(table)
            if len(fkeys) > 0:
                for key in fkeys:
                    # Append Cell class with fkey name, color of parent table
                    columns.append(
                        Cell(
                            value=key.column_name,
                            color=colors_dict[key.parent_table_name],
                        )
                    )

            # Add in the remaining fields
            report = self.eval_dict[table].individual_report_json
            for item in report["fields"]:
                # Append Cell class with name of field, no formatting
                columns.append(Cell(value=item["name"]))

            # Add columns to columns_data array
            columns_data.append(columns)
        # Convert columns_data to dataframe, transpose, and fill N/As
        cols_df = pd.DataFrame(columns_data, index=tables).T.fillna("")
        # Convert df to dictionary
        fields_dict = cols_df.to_dict("split")

        return fields_dict

    def render(self):
        out = None
        out_path = f"{self.dir}/multitable-report.html"
        try:
            out = template.render(report=self)
            try:
                with open(out_path, "w") as f:
                    f.write(out)
            except Exception:
                logger.exception("Failed to save report.")
        except Exception:
            logger.exception("Failed to render report.")
        return out


@dataclass
class Cell:
    """
    Dataclass to hold the formatting properties of the cell for rendering Table Relationships table

    Attributes:
        value: str -  The value of the cell (in this case, the field name)
        color: str -  Hexcode to use for background-color of cell, related keys are displayed with same color
        bold: str - 'bold' if value is a primary key, else empty string
    """

    value: str
    color: str = ""
    bold: str = ""


def assign_key_colors(rel_data):
    colors = ["#ABEBC6", "#ABC6EB", "#EBABC6", "#EBC6AB", "#C6ABEB", "#C6EBAB"]
    color_dict = {}
    i = 0
    for table in rel_data:
        if rel_data[table]["primary_key"] is not None:
            color_dict[table] = colors[i]
            if i < len(colors) - 1:
                i = i + 1
            else:
                i = 0
        else:
            color_dict[table] = ""
    return color_dict
