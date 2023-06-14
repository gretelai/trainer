import re
from typing import Optional

import pandas as pd

from gretel_trainer.relational.core import ForeignKey, RelationalData

_START_LINEAGE = "self"
_GEN_DELIMITER = "."
_COL_DELIMITER = "+"
_END_LINEAGE = "|"


def get_multigenerational_primary_key(
    rel_data: RelationalData, table: str
) -> list[str]:
    "Returns the provided table's primary key with the ancestral lineage prefix appended"
    return [
        f"{_START_LINEAGE}{_END_LINEAGE}{pk}" for pk in rel_data.get_primary_key(table)
    ]


def get_ancestral_foreign_key_maps(
    rel_data: RelationalData, table: str
) -> list[tuple[str, str]]:
    """
    Returns a list of two-element tuples where the first element is a foreign key column
    with ancestral lineage prefix, and the second element is the ancestral-lineage-prefixed
    referred column. This function ultimately provides a list of which columns are duplicates
    in a fully-joined ancestral table (i.e. `get_table_data_with_ancestors`) (only between
    the provided table and its direct parents, not between parents and grandparents).

    For example: given an events table with foreign key `events.user_id` => `users.id`,
    this method returns: [("self|user_id", "self.user_id|id")]
    """

    def _ancestral_fk_map(fk: ForeignKey) -> list[tuple[str, str]]:
        maps = []
        fk_lineage = _COL_DELIMITER.join(fk.columns)

        for i in range(len(fk.columns)):
            fk_col = fk.columns[i]
            ref_col = fk.parent_columns[i]

            maps.append(
                (
                    f"{_START_LINEAGE}{_END_LINEAGE}{fk_col}",
                    f"{_START_LINEAGE}{_GEN_DELIMITER}{fk_lineage}{_END_LINEAGE}{ref_col}",
                )
            )

        return maps

    return [
        fkmap
        for fk in rel_data.get_foreign_keys(table)
        for fkmap in _ancestral_fk_map(fk)
    ]


def get_seed_safe_multigenerational_columns(
    rel_data: RelationalData,
) -> dict[str, list[str]]:
    """
    Returns a dict with Scope.MODELABLE table names as keys and lists of columns to use
    for conditional seeding as values. By using a tableset of empty dataframes, this provides
    a significantly faster / less resource-intensive way to get just the column names
    from the results of `get_table_data_with_ancestors` for all tables.
    """
    tableset = {
        table: pd.DataFrame(columns=list(rel_data.get_table_columns(table)))
        for table in rel_data.list_all_tables()
    }
    return {
        table: list(
            get_table_data_with_ancestors(
                rel_data, table, tableset, ancestral_seeding=True
            ).columns
        )
        for table in rel_data.list_all_tables()
    }


def get_table_data_with_ancestors(
    rel_data: RelationalData,
    table: str,
    tableset: Optional[dict[str, pd.DataFrame]] = None,
    ancestral_seeding: bool = False,
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
    return _join_parents(rel_data, df, table, lineage, tableset, ancestral_seeding)


def _join_parents(
    rel_data: RelationalData,
    df: pd.DataFrame,
    table: str,
    lineage: str,
    tableset: Optional[dict[str, pd.DataFrame]],
    ancestral_seeding: bool,
) -> pd.DataFrame:
    for foreign_key in rel_data.get_foreign_keys(table):
        fk_lineage = _COL_DELIMITER.join(foreign_key.columns)
        next_lineage = f"{lineage}{_GEN_DELIMITER}{fk_lineage}"

        parent_table_name = foreign_key.parent_table_name

        if ancestral_seeding:
            usecols = list(rel_data.get_safe_ancestral_seed_columns(parent_table_name))
        else:
            usecols = rel_data.get_table_columns(parent_table_name)

        if tableset is not None:
            parent_data = tableset[parent_table_name][list(usecols)]
        else:
            parent_data = rel_data.get_table_data(parent_table_name, usecols=usecols)

        df = df.merge(
            parent_data.add_prefix(f"{next_lineage}{_END_LINEAGE}"),
            how="left",
            left_on=[f"{lineage}{_END_LINEAGE}{col}" for col in foreign_key.columns],
            right_on=[
                f"{next_lineage}{_END_LINEAGE}{parent_col}"
                for parent_col in foreign_key.parent_columns
            ],
        )

        df = _join_parents(
            rel_data, df, parent_table_name, next_lineage, tableset, ancestral_seeding
        )
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


def prepend_foreign_key_lineage(df: pd.DataFrame, fk_cols: list[str]) -> pd.DataFrame:
    """
    Given a multigenerational dataframe, renames all columns such that the provided
    foreign key columns act as the lineage from some child table to the provided data.
    The resulting column names are elder-generation ancestral column names from the
    perspective of a child table that relates to that parent via the provided foreign key.
    """
    fk_lineage = _COL_DELIMITER.join(fk_cols)

    def _adjust(col: str) -> str:
        return col.replace(
            _START_LINEAGE,
            f"{_START_LINEAGE}{_GEN_DELIMITER}{fk_lineage}",
            1,
        )

    mapper = {col: _adjust(col) for col in df.columns}
    return df.rename(columns=mapper)
