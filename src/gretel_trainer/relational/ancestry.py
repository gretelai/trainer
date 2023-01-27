import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

from gretel_trainer.relational.core import ForeignKey, RelationalData

_START_LINEAGE = "self"
_GEN_DELIMITER = "."
_END_LINEAGE = "|"


def get_multigenerational_primary_key(
    rel_data: RelationalData, table: str
) -> Optional[str]:
    pk = rel_data.get_primary_key(table)
    if pk is None:
        return None
    else:
        return f"{_START_LINEAGE}{_END_LINEAGE}{pk}"


def get_ancestral_foreign_key_maps(
    rel_data: RelationalData, table: str
) -> List[Tuple[str, str]]:
    def _ancestral_fk_map(fk: ForeignKey) -> Tuple[str, str]:
        fk_col = fk.column_name
        ref_col = fk.parent_column_name

        ancestral_foreign_key = f"{_START_LINEAGE}{_END_LINEAGE}{fk_col}"
        ancestral_referenced_col = (
            f"{_START_LINEAGE}{_GEN_DELIMITER}{fk_col}{_END_LINEAGE}{ref_col}"
        )

        return (ancestral_foreign_key, ancestral_referenced_col)

    return [_ancestral_fk_map(fk) for fk in rel_data.get_foreign_keys(table)]


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
        next_lineage = f"{lineage}{_GEN_DELIMITER}{foreign_key.column_name}"

        parent_table_name = foreign_key.parent_table_name
        if tableset is not None:
            parent_data = tableset[parent_table_name]
        else:
            parent_data = rel_data.get_table_data(parent_table_name)
        parent_data = parent_data.add_prefix(f"{next_lineage}{_END_LINEAGE}")

        df = df.merge(
            parent_data,
            how="left",
            left_on=f"{lineage}{_END_LINEAGE}{foreign_key.column_name}",
            right_on=f"{next_lineage}{_END_LINEAGE}{foreign_key.parent_column_name}",
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
        col: _removeprefix(col, f"{_START_LINEAGE}{_END_LINEAGE}")
        for col in root_columns
    }
    return df[root_columns].rename(columns=mapper)


def prepend_foreign_key_lineage(df: pd.DataFrame, fk_col: str) -> pd.DataFrame:
    """
    Given a multigenerational dataframe, renames all columns such that the provided
    foreign key acts as the lineage from some child table to the provided data.
    The resulting column names are elder-generation ancestral column names from the
    perspective of a child table that relates to that parent via the provided foreign key.
    """

    def _adjust(col: str) -> str:
        return col.replace(
            _START_LINEAGE,
            f"{_START_LINEAGE}{_GEN_DELIMITER}{fk_col}",
            1,
        )

    mapper = {col: _adjust(col) for col in df.columns}
    return df.rename(columns=mapper)


def _removeprefix(s: str, prefix: str) -> str:
    if s.startswith(prefix):
        return s[len(prefix) :]
    else:
        return s
