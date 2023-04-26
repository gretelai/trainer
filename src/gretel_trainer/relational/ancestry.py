import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

from gretel_trainer.relational.core import ForeignKey, RelationalData

_START_LINEAGE = "self"
_GEN_DELIMITER = "."
_COL_DELIMITER = "+"
_END_LINEAGE = "|"


def get_multigenerational_primary_key(
    rel_data: RelationalData, table: str
) -> List[str]:
    return [
        f"{_START_LINEAGE}{_END_LINEAGE}{pk}" for pk in rel_data.get_primary_key(table)
    ]


def get_ancestral_foreign_key_maps(
    rel_data: RelationalData, table: str
) -> List[Tuple[str, str]]:
    def _ancestral_fk_map(fk: ForeignKey) -> List[Tuple[str, str]]:
        maps = []
        fk_columns = _COL_DELIMITER.join(fk.columns)

        for i in range(len(fk.columns)):
            fk_col = fk.columns[i]
            ref_col = fk.parent_columns[i]

            maps.append(
                (
                    f"{_START_LINEAGE}{_END_LINEAGE}{fk_col}",
                    f"{_START_LINEAGE}{_GEN_DELIMITER}{fk_columns}{_END_LINEAGE}{ref_col}",
                )
            )

        return maps

    return [
        fkmap
        for fk in rel_data.get_foreign_keys(table)
        for fkmap in _ancestral_fk_map(fk)
    ]


def get_table_data_with_ancestors(
    rel_data: RelationalData,
    table: str,
    tableset: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """
    Returns a data frame with all ancestral data joined to each record.
    Column names are modified to the format `LINAGE|COLUMN_NAME`.
    Lineage begins with `self` for the supplied `table`, and as older
    generations are joined, the foreign keys to those generations are appended,
    separated by periods.

    If `tableset` is provided, use it in place of the source data in `self.graph`.
    """
    lineage = _START_LINEAGE
    if tableset is not None:
        df = tableset[table]
    else:
        df = rel_data.get_table_data(table)
    df = df.add_prefix(f"{_START_LINEAGE}{_END_LINEAGE}")
    return _join_parents(rel_data, df, table, lineage, tableset)


def _join_parents(
    rel_data: RelationalData,
    df: pd.DataFrame,
    table: str,
    lineage: str,
    tableset: Optional[Dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    for foreign_key in rel_data.get_foreign_keys(table):
        fk_col = _COL_DELIMITER.join(foreign_key.columns)
        next_lineage = f"{lineage}{_GEN_DELIMITER}{fk_col}"

        parent_table_name = foreign_key.parent_table_name
        if tableset is not None:
            parent_data = tableset[parent_table_name]
        else:
            parent_data = rel_data.get_table_data(parent_table_name)
        parent_data = parent_data.add_prefix(f"{next_lineage}{_END_LINEAGE}")

        df = df.merge(
            parent_data,
            how="left",
            left_on=[f"{lineage}{_END_LINEAGE}{col}" for col in foreign_key.columns],
            right_on=[
                f"{next_lineage}{_END_LINEAGE}{parent_col}"
                for parent_col in foreign_key.parent_columns
            ],
        )

        df = _join_parents(rel_data, df, parent_table_name, next_lineage, tableset)
    return df


def is_ancestral_column(column: str) -> bool:
    """
    Returns True if the provided column name corresponds to an elder-generation ancestor.
    """
    regex_string = rf"\{_GEN_DELIMITER}[^\{_END_LINEAGE}]+\{_END_LINEAGE}"
    regex = re.compile(regex_string)
    return bool(regex.search(column))


def drop_ancestral_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops ancestral columns from the given dataframe and removes the lineage prefix
    from the remaining columns, restoring them to their original source names.
    """
    root_columns = [
        col for col in df.columns if col.startswith(f"{_START_LINEAGE}{_END_LINEAGE}")
    ]
    mapper = {
        col: col.removeprefix(f"{_START_LINEAGE}{_END_LINEAGE}") for col in root_columns
    }
    return df[root_columns].rename(columns=mapper)


def prepend_foreign_key_lineage(df: pd.DataFrame, fk_cols: List[str]) -> pd.DataFrame:
    """
    Given a multigenerational dataframe, renames all columns such that the provided
    foreign key columns act as the lineage from some child table to the provided data.
    The resulting column names are elder-generation ancestral column names from the
    perspective of a child table that relates to that parent via the provided foreign key.
    """
    fk = _COL_DELIMITER.join(fk_cols)

    def _adjust(col: str) -> str:
        return col.replace(
            _START_LINEAGE,
            f"{_START_LINEAGE}{_GEN_DELIMITER}{fk}",
            1,
        )

    mapper = {col: _adjust(col) for col in df.columns}
    return df.rename(columns=mapper)
