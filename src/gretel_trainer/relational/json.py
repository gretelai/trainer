from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from json import JSONDecodeError, dumps, loads
from typing import Any, Optional, Protocol, Union
from uuid import uuid4

import numpy as np
import pandas as pd
from unflatten import unflatten

logger = logging.getLogger(__name__)

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
    nested_dfs: list[tuple[str, pd.DataFrame]],
    flat_dfs: list[tuple[str, pd.DataFrame]],
    columns: Optional[list[str]] = None,
) -> list[tuple[str, pd.DataFrame]]:
    if not nested_dfs:
        return flat_dfs
    name, df = nested_dfs.pop()
    cols_to_scan = columns or [col for col in df.columns if df.dtypes[col] == "object"]
    dict_cols = [col for col in cols_to_scan if df[col].dropna().apply(is_dict).all()]
    if dict_cols:
        df[dict_cols] = nulls_to_empty_dicts(df[dict_cols])
        for col in dict_cols:
            new_cols = pandas_json_normalize(df[col]).add_prefix(col + FIELD_SEPARATOR)
            df = pd.concat([df, new_cols], axis="columns")
            df = df.drop(columns=new_cols.columns[new_cols.isnull().all()])
        nested_dfs.append((name, df.drop(columns=dict_cols)))
    else:
        list_cols = [
            col for col in cols_to_scan if df[col].dropna().apply(is_list).all()
        ]
        if list_cols:
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


def generate_unique_table_name(s: str):
    sanitized_str = "-".join(re.findall(r"[a-zA-Z_0-9]+", s))
    # Generate unique suffix to prevent collisions
    unique_suffix = make_suffix()
    # Max length for a table/filename is 128 chars
    return f"{sanitized_str[:80]}_invented_{unique_suffix}"


def make_suffix():
    return uuid4().hex


def jsonencode(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Returns a dataframe with the specified columns transformed such that their JSON-like
    values can be written to CSV and later re-read back to Pandas from CSV.
    """
    # Save memory and return the *original dataframe* (not a copy!) if no columns to transform
    if len(cols) == 0:
        return df

    def _jsonencode(x):
        if isinstance(x, str):
            return x
        elif isinstance(x, (dict, list)):
            return dumps(x)

    copy = df.copy()
    for col in cols:
        copy[col] = copy[col].map(_jsonencode)

    return copy


def jsondecode(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Returns a dataframe with the specified columns parsed from JSON to Python objects.
    """
    # Save memory and return the *original dataframe* (not a copy!) if no columns to transform
    if len(cols) == 0:
        return df

    def _jsondecode(obj):
        try:
            return loads(obj)
        except (ValueError, TypeError, JSONDecodeError):
            return obj

    copy = df.copy()
    for col in cols:
        copy[col] = copy[col].map(_jsondecode)

    return copy


class _RelationalData(Protocol):
    def get_foreign_keys(
        self, table: str
    ) -> list:  # can't specify element type (ForeignKey) without cyclic dependency
        ...

    def get_table_columns(self, table: str) -> list[str]:
        ...

    def get_invented_table_metadata(
        self, table: str
    ) -> Optional[InventedTableMetadata]:
        ...


@dataclass
class InventedTableMetadata:
    invented_root_table_name: str
    original_table_name: str
    json_breadcrumb_path: str
    empty: bool


@dataclass
class ProducerMetadata:
    invented_root_table_name: str
    table_name_mappings: dict[str, str]

    @property
    def table_names(self) -> list[str]:
        return list(self.table_name_mappings.values())


def ingest(
    table_name: str,
    primary_key: list[str],
    df: pd.DataFrame,
    json_columns: list[str],
) -> Optional[IngestResponseT]:
    json_decoded = jsondecode(df, json_columns)
    tables = _normalize_json([(table_name, json_decoded)], [], json_columns)

    # If we created additional tables (from JSON lists) or added columns (from JSON dicts)
    if len(tables) > 1 or len(tables[0][1].columns) > len(df.columns):
        # Map json breadcrumbs to uniquely generated table name
        mappings = {name: generate_unique_table_name(table_name) for name, _ in tables}
        logger.info(f"Transformed JSON into {len(mappings)} tables for modeling.")
        logger.debug(f"Invented table names: {list(mappings.values())}")
        commands = _generate_commands(
            tables=tables,
            table_name_mappings=mappings,
            original_table_name=table_name,
            original_primary_key=primary_key,
        )
        producer_metadata = ProducerMetadata(
            invented_root_table_name=mappings[table_name],
            table_name_mappings=mappings,
        )
        return (commands, producer_metadata)


def _generate_commands(
    tables: list[tuple[str, pd.DataFrame]],
    table_name_mappings: dict[str, str],
    original_table_name: str,
    original_primary_key: list[str],
) -> CommandsT:
    """
    Returns lists of keyword arguments designed to be passed to a
    RelationalData instance's _add_single_table and add_foreign_key methods
    """
    root_table_name = table_name_mappings[original_table_name]

    _add_single_table = []
    add_foreign_key = []

    for table_breadcrumb_name, table_df in tables:
        table_name = table_name_mappings[table_breadcrumb_name]
        if table_name == root_table_name:
            table_pk = original_primary_key + [PRIMARY_KEY_COLUMN]
        else:
            table_pk = [PRIMARY_KEY_COLUMN]
        table_df.index.rename(PRIMARY_KEY_COLUMN, inplace=True)
        table_df.reset_index(inplace=True)
        metadata = InventedTableMetadata(
            invented_root_table_name=root_table_name,
            original_table_name=original_table_name,
            json_breadcrumb_path=table_breadcrumb_name,
            empty=table_df.empty,
        )
        _add_single_table.append(
            {
                "name": table_name,
                "primary_key": table_pk,
                "data": table_df,
                "invented_table_metadata": metadata,
            }
        )

    for table_breadcrumb_name, table_df in tables:
        table_name = table_name_mappings[table_breadcrumb_name]
        for column in get_id_columns(table_df):
            referred_table = table_name_mappings[
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


def restore(
    tables: dict[str, pd.DataFrame],
    rel_data: _RelationalData,
    root_table_name: str,
    original_columns: list[str],
    table_name_mappings: dict[str, str],
    original_table_name: str,
) -> Optional[pd.DataFrame]:
    # If the root invented table is not present, we are completely out of luck
    # (Missing invented child tables can be replaced with empty lists so we at least provide _something_)
    if root_table_name not in tables:
        logger.warning(
            f"Cannot restore nested JSON data: root invented table `{root_table_name}` is missing from output tables."
        )
        return None

    return _denormalize_json(
        tables, rel_data, table_name_mappings, original_table_name
    )[original_columns]


def _denormalize_json(
    tables: dict[str, pd.DataFrame],
    rel_data: _RelationalData,
    table_name_mappings: dict[str, str],
    original_table_name: str,
) -> pd.DataFrame:
    table_names = list(table_name_mappings.values())
    inverse_table_name_mappings = {v: k for k, v in table_name_mappings.items()}
    table_dict: dict = {inverse_table_name_mappings[k]: v for k, v in tables.items()}
    for table_name in list(reversed(table_names)):
        table_provenance_name = inverse_table_name_mappings[table_name]
        empty_fallback = pd.DataFrame(
            data={col: [] for col in rel_data.get_table_columns(table_name)},
        )
        table_df = table_dict.get(table_provenance_name, empty_fallback)

        if table_df.empty and _is_invented_child_table(table_name, rel_data):
            p_name = rel_data.get_foreign_keys(table_name)[0].parent_table_name
            parent_name = inverse_table_name_mappings[p_name]
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
                parent_name = inverse_table_name_mappings[p_name]
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
    return table_dict[original_table_name]


def get_json_columns(df: pd.DataFrame) -> list[str]:
    """
    Samples non-null values from all columns and returns those that contain
    valid JSON dicts or lists.

    Raises an error if *all* columns are lists, as that is not currently supported.
    """
    object_cols = {
        col: data
        for col in df.columns
        if df.dtypes[col] == "object" and len(data := df[col].dropna()) > 0
    }

    if len(object_cols) == 0:
        return []

    list_cols = [
        col for col, series in object_cols.items() if series.apply(is_list).all()
    ]

    if len(list_cols) == len(df.columns):
        raise ValueError("Cannot accept tables with JSON lists in all columns")

    dict_cols = [
        col
        for col, series in object_cols.items()
        if col not in list_cols and series.apply(is_dict).all()
    ]

    return dict_cols + list_cols


CommandsT = tuple[list[dict], list[dict]]
IngestResponseT = tuple[CommandsT, ProducerMetadata]
