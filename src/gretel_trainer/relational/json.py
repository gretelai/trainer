# @title Define functions for JSON to/from RDB conversion

import hashlib
import re
from json import JSONDecodeError, loads

import numpy as np
import pandas as pd
from unflatten import unflatten

from gretel_trainer.relational import MultiTable, RelationalData

# JSON dict to multi-column and list to multi-table

FIELD_SEPARATOR = ">"
TABLE_SEPARATOR = "^"
ID_SUFFIX = "~id"
ORDER_COLUMN = "array~order"
CONTENT_COLUMN = "content"
INPUT_TABLE = "_ROOT_"
PRIMARY_KEY_COLUMN = "~PRIMARY_KEY_ID~"


def load_json(obj):
    if isinstance(obj, (dict, list)):
        return obj
    else:
        return loads(obj)


def is_json(obj, json_type=(dict, list)):
    try:
        obj = load_json(obj)
    except (ValueError, TypeError, JSONDecodeError):
        return False
    else:
        return isinstance(obj, json_type)


def is_dict(obj):
    return is_json(obj, dict)


def is_list(obj):
    return is_json(obj, list)


def pandas_json_normalize(df):
    return pd.json_normalize(df.apply(load_json).to_list(), sep=FIELD_SEPARATOR)


def nulls_to_empty_dicts(df):
    return df.applymap(lambda x: {} if pd.isnull(x) else x)


def nulls_to_empty_lists(series):
    return series.apply(lambda x: x if isinstance(x, list) or not pd.isnull(x) else [])


def _normalize_json(nested_dfs, flat_dfs=None):
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


def normalize_json(df):
    return _normalize_json([(INPUT_TABLE, df.copy())], [])


# Multi-table and multi-column back to single-table with JSON


def get_id_columns(df):
    return [col for col in df.columns if col.endswith(ID_SUFFIX)]


def get_parent_table_name_from_child_id_column(id_column_name):
    return id_column_name[: -len(ID_SUFFIX)]


def get_parent_column_name_from_child_table_name(table_name):
    return table_name.split(TABLE_SEPARATOR)[-1]


def is_child_table(df):
    id_columns = get_id_columns(df)
    if len(id_columns) != 1:
        return False
    id_col = id_columns[0]
    return all([col in df.columns for col in (id_col, ORDER_COLUMN, CONTENT_COLUMN)])


def denormalize_json(flat_tables):
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
    return table_dict[INPUT_TABLE]


# Create foreign to primary key mapping for Gretel Relational


def sanitize_str(s):
    sanitized_str = "-".join(re.findall(r"[a-zA-Z_0-9]+", s))
    # Generate suffix from original string, in case of sanitized_str collision
    unique_suffix = hashlib.sha256(s.encode("utf-8")).hexdigest()[:10]
    return f"{sanitized_str}-{unique_suffix}"


class MockMultiTable:
    def __init__(self, flat_tables):
        self.synthetic_output_tables = dict(flat_tables)


class GretelJSON:
    def __init__(self, df, debug=False, **kwargs):
        self.debug = debug

        self.columns = df.columns
        tables = normalize_json(df)
        self.table_name_mappings = {name: sanitize_str(name) for name, _ in tables}
        tables = [(self.table_name_mappings[name], df) for name, df in tables]
        self.tables = [table_name for table_name, _ in tables]
        self.empty_tables = dict([t for t in tables if t[1].empty])
        non_empty_tables = [t for t in tables if t[0] not in self.empty_tables.keys()]

        relational_data = RelationalData()

        for table_name, table_df in non_empty_tables:
            table_df.index.rename(PRIMARY_KEY_COLUMN, inplace=True)
            relational_data.add_table(
                name=table_name,
                primary_key=PRIMARY_KEY_COLUMN,
                data=table_df.reset_index(),
            )

        for table_name, table_df in non_empty_tables:
            for column in get_id_columns(table_df):
                fk = table_name + "." + column
                ref = (
                    self.table_name_mappings[
                        get_parent_table_name_from_child_id_column(column)
                    ]
                    + "."
                    + PRIMARY_KEY_COLUMN
                )
                relational_data.add_foreign_key(foreign_key=fk, referencing=ref)

        if self.debug:
            self.multitable = MockMultiTable(non_empty_tables)
        else:
            self.multitable = MultiTable(relational_data, **kwargs)

    def train(self):
        if not self.debug:
            self.multitable.train()

    def generate(self, **kwargs):
        if not self.debug:
            self.multitable.generate(**kwargs)
        return self.get_generated_tables()

    def get_generated_tables(self):
        output_tables = []
        inverse_table_name_mappings = {
            value: key for key, value in self.table_name_mappings.items()
        }
        for t in self.tables:
            empty_table = self.empty_tables.get(t)
            multitable_output = self.multitable.synthetic_output_tables.get(
                t, empty_table
            )
            output_tables.append((inverse_table_name_mappings[t], multitable_output))
        return denormalize_json(output_tables)[self.columns]

    def save_jsonl(self, filename):
        return self.get_generated_tables().to_json(
            filename, orient="records", lines=True
        )

    def display_report(self):
        import IPython
        from smart_open import open

        report_path = str(
            self.multitable._working_dir
            / self.multitable._synthetics_run.identifier
            / "relational_report.html"
        )
        IPython.display.display(IPython.display.HTML(data=open(report_path).read()))
