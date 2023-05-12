from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from json import JSONDecodeError, loads
from typing import Any, Optional, Protocol, Union

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


def _is_invented_child_table(table: str, rel_data: _RelationalData) -> bool:
    imeta = rel_data.get_invented_table_metadata(table)
    return imeta is not None and imeta.invented_root_table_name != table


def sanitize_str(s):
    sanitized_str = "-".join(re.findall(r"[a-zA-Z_0-9]+", s))
    # Generate suffix from original string, in case of sanitized_str collision
    unique_suffix = make_suffix(s)
    return f"{sanitized_str}-{unique_suffix}"


def make_suffix(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:10]


class _RelationalData(Protocol):
    def get_foreign_keys(
        self, table: str
    ) -> list:  # can't specify element type (ForeignKey) without cyclic dependency
        ...

    def get_table_columns(self, table: str) -> set[str]:
        ...

    def get_invented_table_metadata(
        self, table: str
    ) -> Optional[InventedTableMetadata]:
        ...


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
        table_name_mappings: list[tuple[str, str]],
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
            mappings = [(name, sanitize_str(name)) for name, _ in tables]
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
        return self._mapping_dict[self.original_table_name]

    @property
    def table_names(self) -> list[str]:
        # We need to keep the order intact for restoring
        return [m[1] for m in self.table_name_mappings]

    def get_sanitized_name(self, t: str) -> str:
        return self._mapping_dict[t]

    @property
    def _mapping_dict(self) -> dict[str, str]:
        return dict(self.table_name_mappings)

    @property
    def inverse_table_name_mappings(self) -> dict[str, str]:
        # Keys are sanitized, model-friendly names
        # Values are "provenance" names (a^b>c) or the original table name
        return {value: key for key, value in self._mapping_dict.items()}

    def restore(
        self, tables: dict[str, pd.DataFrame], rel_data: _RelationalData
    ) -> Optional[pd.DataFrame]:
        """Reduces a set of tables (assumed to match the shapes created on initialization)
        to a single table matching the shape of the original source table
        """
        # If the root invented table failed, we are completely out of luck
        # (Missing invented child tables can be replaced with empty lists so we at least provide _something_)
        if self.root_table_name not in tables:
            return None

        return self._denormalize_json(tables, rel_data)[self.original_columns]

    def _denormalize_json(
        self, tables: dict[str, pd.DataFrame], rel_data: _RelationalData
    ) -> pd.DataFrame:
        table_dict: dict = {
            self.inverse_table_name_mappings[k]: v for k, v in tables.items()
        }
        for table_name in list(reversed(self.table_names)):
            table_provenance_name = self.inverse_table_name_mappings[table_name]
            empty_fallback = pd.DataFrame(
                data={col: [] for col in rel_data.get_table_columns(table_name)},
            )
            table_df = table_dict.get(table_provenance_name, empty_fallback)

            if table_df.empty and _is_invented_child_table(table_name, rel_data):
                p_name = rel_data.get_foreign_keys(table_name)[0].parent_table_name
                parent_name = self.inverse_table_name_mappings[p_name]
                col_name = get_parent_column_name_from_child_table_name(
                    table_provenance_name
                )
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
                if _is_invented_child_table(table_name, rel_data):
                    # we know there is exactly one foreign key on invented child tables...
                    fk = rel_data.get_foreign_keys(table_name)[0]
                    # ...with exactly one column
                    fk_col = fk.columns[0]
                    p_name = fk.parent_table_name
                    parent_name = self.inverse_table_name_mappings[p_name]
                    nested_df = (
                        nested_df.sort_values(ORDER_COLUMN)
                        .groupby(fk_col)[CONTENT_COLUMN]
                        .agg(list)
                    )
                    col_name = get_parent_column_name_from_child_table_name(
                        table_provenance_name
                    )
                    table_dict[parent_name] = table_dict[parent_name].join(
                        nested_df.rename(col_name)
                    )
                    table_dict[parent_name][col_name] = nulls_to_empty_lists(
                        table_dict[parent_name][col_name]
                    )
                table_dict[table_provenance_name] = nested_df
        return table_dict[self.original_table_name]


def _generate_commands(
    rel_json: RelationalJson, tables: list[tuple[str, pd.DataFrame]]
) -> CommandsT:
    """
    Returns lists of keyword arguments designed to be passed to a
    RelationalData instance's _add_single_table and add_foreign_key methods
    """
    tables = [(rel_json.get_sanitized_name(name), df) for name, df in tables]
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
        invented_root_table_name = rel_json.get_sanitized_name(
            rel_json.original_table_name
        )
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
            referred_table = rel_json.get_sanitized_name(
                get_parent_table_name_from_child_id_column(column)
            )
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
