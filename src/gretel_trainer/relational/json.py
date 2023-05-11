from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from json import JSONDecodeError, loads
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from unflatten import unflatten

# JSON dict to multi-column and list to multi-table

FIELD_SEPARATOR = ">"
TABLE_SEPARATOR = "^"
ID_SUFFIX = "~id"
ORDER_COLUMN = "array~order"
CONTENT_COLUMN = "content"
PRIMARY_KEY_COLUMN = "~PRIMARY_KEY_ID~"


def load_json(obj: Any) -> Union[dict, list]:
    if isinstance(obj, (dict, list)):
        return obj
    else:
        return loads(obj)


def is_json(obj: Any, json_type=(dict, list)) -> bool:
    try:
        obj = load_json(obj)
    except (ValueError, TypeError, JSONDecodeError):
        return False
    else:
        return isinstance(obj, json_type)


def is_dict(obj: Any) -> bool:
    return is_json(obj, dict)


def is_list(obj: Any) -> bool:
    return isinstance(obj, np.ndarray) or is_json(obj, list)


def pandas_json_normalize(series: pd.Series) -> pd.DataFrame:
    return pd.json_normalize(series.apply(load_json).to_list(), sep=FIELD_SEPARATOR)


def nulls_to_empty_dicts(df: pd.DataFrame) -> pd.DataFrame:
    return df.applymap(lambda x: {} if pd.isnull(x) else x)


def nulls_to_empty_lists(series: pd.Series) -> pd.Series:
    return series.apply(lambda x: x if isinstance(x, list) or not pd.isnull(x) else [])


def _normalize_json(
    nested_dfs: list[tuple[str, pd.DataFrame]], flat_dfs: list[tuple[str, pd.DataFrame]]
) -> list[tuple[str, pd.DataFrame]]:
    if not nested_dfs:
        return flat_dfs
    name, df = nested_dfs.pop()
    dict_cols = [
        col
        for col in df.columns
        if df[col].apply(is_dict).any() and df[col].dropna().apply(is_dict).all()
    ]
    list_cols = [
        col
        for col in df.columns
        if df[col].apply(is_list).any() and df[col].dropna().apply(is_list).all()
    ]
    if dict_cols:
        df[dict_cols] = nulls_to_empty_dicts(df[dict_cols])
        for col in dict_cols:
            new_cols = pandas_json_normalize(df[col]).add_prefix(col + FIELD_SEPARATOR)
            df = pd.concat([df, new_cols], axis="columns")
            df = df.drop(columns=new_cols.columns[new_cols.isnull().all()])
        nested_dfs.append((name, df.drop(columns=dict_cols)))
    elif list_cols:
        for col in list_cols:
            new_table = df[col].explode().dropna().rename(CONTENT_COLUMN).to_frame()
            new_table[ORDER_COLUMN] = new_table.groupby(level=0).cumcount()
            nested_dfs.append(
                (
                    name + TABLE_SEPARATOR + col,
                    new_table.reset_index(names=name + ID_SUFFIX),
                )
            )
        nested_dfs.append((name, df.drop(columns=list_cols)))
    else:
        flat_dfs.append((name, df))
    return _normalize_json(nested_dfs, flat_dfs)


# Multi-table and multi-column back to single-table with JSON


def get_id_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.endswith(ID_SUFFIX)]


def get_parent_table_name_from_child_id_column(id_column_name: str) -> str:
    return id_column_name[: -len(ID_SUFFIX)]


def get_parent_column_name_from_child_table_name(table_name: str) -> str:
    return table_name.split(TABLE_SEPARATOR)[-1]


def is_child_table(df: pd.DataFrame) -> bool:
    id_columns = get_id_columns(df)
    if len(id_columns) != 1:
        return False
    id_col = id_columns[0]
    return all([col in df.columns for col in (id_col, ORDER_COLUMN, CONTENT_COLUMN)])


def denormalize_json(
    flat_tables: list[tuple[str, pd.DataFrame]], root_table: str
) -> pd.DataFrame:
    table_dict = dict(reversed(flat_tables))
    for table_name, table_df in table_dict.items():
        if table_df.empty and is_child_table(table_df):
            id_col = get_id_columns(table_df)[0]
            parent_name = get_parent_table_name_from_child_id_column(id_col)
            col_name = get_parent_column_name_from_child_table_name(table_name)
            kwargs = {col_name: table_dict[parent_name].apply(lambda x: [], axis=1)}
            table_dict[parent_name] = table_dict[parent_name].assign(**kwargs)
        else:
            col_names = [col for col in table_df.columns if FIELD_SEPARATOR in col]
            new_col_names = [col.replace(FIELD_SEPARATOR, ".") for col in col_names]
            flat_df = table_df[col_names].rename(
                columns=dict(zip(col_names, new_col_names))
            )
            flat_dict = {
                k: {
                    k1: v1
                    for k1, v1 in v.items()
                    if v1 is not np.nan and v1 is not None
                }
                for k, v in flat_df.to_dict("index").items()
            }
            dict_df = nulls_to_empty_dicts(
                pd.DataFrame.from_dict(
                    {k: unflatten(v) for k, v in flat_dict.items()}, orient="index"
                )
            )
            nested_df = table_df.join(dict_df).drop(columns=col_names)
            if is_child_table(nested_df):
                id_col = get_id_columns(nested_df)[0]
                parent_name = get_parent_table_name_from_child_id_column(id_col)
                nested_df = (
                    nested_df.sort_values(ORDER_COLUMN)
                    .groupby(id_col)[CONTENT_COLUMN]
                    .agg(list)
                )
                col_name = get_parent_column_name_from_child_table_name(table_name)
                table_dict[parent_name] = table_dict[parent_name].join(
                    nested_df.rename(col_name)
                )
                table_dict[parent_name][col_name] = nulls_to_empty_lists(
                    table_dict[parent_name][col_name]
                )
            table_dict[table_name] = nested_df
    return table_dict[root_table]


def sanitize_str(s):
    sanitized_str = "-".join(re.findall(r"[a-zA-Z_0-9]+", s))
    # Generate suffix from original string, in case of sanitized_str collision
    unique_suffix = make_suffix(s)
    return f"{sanitized_str}-{unique_suffix}"


def make_suffix(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:10]


@dataclass
class InventedTableMetadata:
    invented_root_table_name: str
    original_table_name: str


class RelationalJson:
    def __init__(
        self,
        original_table_name: str,
        original_primary_key: list[str],
        original_columns: list[str],
        original_data: Optional[pd.DataFrame],
        table_name_mappings: dict[str, str],
    ):
        self.original_table_name = original_table_name
        self.original_primary_key = original_primary_key
        self.original_columns = original_columns
        self.original_data = original_data
        self.table_name_mappings = table_name_mappings

    @classmethod
    def ingest(
        cls, table_name: str, primary_key: list[str], df: pd.DataFrame
    ) -> Optional[IngestResponseT]:
        tables = _normalize_json([(table_name, df.copy())], [])
        # If we created additional tables (from JSON lists) or added columns (from JSON dicts)
        if len(tables) > 1 or len(tables[0][1].columns) > len(df.columns):
            mappings = {name: sanitize_str(name) for name, _ in tables}
            rel_json = RelationalJson(
                original_table_name=table_name,
                original_primary_key=primary_key,
                original_data=df,
                original_columns=list(df.columns),
                table_name_mappings=mappings,
            )
            commands = _generate_commands(rel_json, tables)
            return (rel_json, commands)

    @property
    def root_table_name(self) -> str:
        return self.table_name_mappings[self.original_table_name]

    @property
    def table_names(self) -> list[str]:
        return list(self.table_name_mappings.values())

    def restore(
        self, tables: dict[str, pd.DataFrame], cols: dict[str, set[str]]
    ) -> Optional[pd.DataFrame]:
        """Reduces a set of tables (assumed to match the shapes created on initialization)
        to a single table matching the shape of the original source table
        """
        output_tables = []
        for t in self.table_names:
            empty_fallback = pd.DataFrame(data={col: [] for col in cols[t]})
            multitable_output = tables.get(t, empty_fallback)

            # If the root invented table failed, we are completely out of luck
            # (Missing invented child tables can be replaced with empty lists so we at least provide _something_)
            if multitable_output.empty and t == self.root_table_name:
                return None

            output_tables.append(
                (self.inverse_table_name_mappings[t], multitable_output)
            )
        return denormalize_json(output_tables, self.original_table_name)[
            self.original_columns
        ]

    @property
    def inverse_table_name_mappings(self) -> dict[str, str]:
        return {value: key for key, value in self.table_name_mappings.items()}


def _generate_commands(
    rel_json: RelationalJson, tables: list[tuple[str, pd.DataFrame]]
) -> CommandsT:
    """
    Returns lists of keyword arguments designed to be passed to a
    RelationalData instance's _add_single_table and add_foreign_key methods
    """
    tables = [(rel_json.table_name_mappings[name], df) for name, df in tables]
    non_empty_tables = [t for t in tables if not t[1].empty]

    _add_single_table = []
    add_foreign_key = []

    for table_name, table_df in non_empty_tables:
        if table_name == rel_json.root_table_name:
            table_pk = rel_json.original_primary_key + [PRIMARY_KEY_COLUMN]
        else:
            table_pk = [PRIMARY_KEY_COLUMN]
        table_df.index.rename(PRIMARY_KEY_COLUMN, inplace=True)
        table_df.reset_index(inplace=True)
        invented_root_table_name = rel_json.table_name_mappings[
            rel_json.original_table_name
        ]
        metadata = InventedTableMetadata(
            invented_root_table_name=invented_root_table_name,
            original_table_name=rel_json.original_table_name,
        )
        _add_single_table.append(
            {
                "name": table_name,
                "primary_key": table_pk,
                "data": table_df,
                "invented_table_metadata": metadata,
            }
        )

    for table_name, table_df in non_empty_tables:
        for column in get_id_columns(table_df):
            referred_table = rel_json.table_name_mappings[
                get_parent_table_name_from_child_id_column(column)
            ]
            add_foreign_key.append(
                {
                    "table": table_name,
                    "constrained_columns": [column],
                    "referred_table": referred_table,
                    "referred_columns": [PRIMARY_KEY_COLUMN],
                }
            )
    return (_add_single_table, add_foreign_key)


CommandsT = tuple[list[dict], list[dict]]
IngestResponseT = tuple[RelationalJson, CommandsT]
