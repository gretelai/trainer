import logging
import random

from typing import Any

import pandas as pd
import smart_open

import gretel_trainer.relational.strategies.common as common

from gretel_trainer.relational.core import GretelModelConfig, RelationalData
from gretel_trainer.relational.output_handler import OutputHandler

logger = logging.getLogger(__name__)


class IndependentStrategy:
    @property
    def name(self) -> str:
        return "independent"

    @property
    def supported_model_keys(self) -> list[str]:
        return ["amplify", "actgan", "synthetics", "tabular_dp"]

    @property
    def default_config(self) -> GretelModelConfig:
        return "synthetics/tabular-actgan"

    def label_encode_keys(
        self, rel_data: RelationalData, tables: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        return common.label_encode_keys(rel_data, tables)

    def prepare_training_data(
        self, rel_data: RelationalData, table_paths: dict[str, str]
    ) -> dict[str, str]:
        """
        Writes tables' training data to provided paths.
        Training data has primary and foreign key columns removed.
        """
        for table, path in table_paths.items():
            columns_to_drop = set()
            columns_to_drop.update(rel_data.get_primary_key(table))
            for foreign_key in rel_data.get_foreign_keys(table):
                columns_to_drop.update(foreign_key.columns)

            all_columns = rel_data.get_table_columns(table)
            use_columns = [col for col in all_columns if col not in columns_to_drop]

            source_path = rel_data.get_table_source(table)
            with smart_open.open(source_path, "rb") as src, smart_open.open(
                path, "wb"
            ) as dest:
                pd.DataFrame(columns=use_columns).to_csv(dest, index=False)
                for chunk in pd.read_csv(src, usecols=use_columns, chunksize=10_000):
                    chunk.to_csv(dest, index=False, mode="a", header=False)

        return table_paths

    def tables_to_retrain(
        self, tables: list[str], rel_data: RelationalData
    ) -> list[str]:
        """
        Returns the provided tables requested to retrain, unaltered.
        """
        return tables

    def validate_preserved_tables(
        self, tables: list[str], rel_data: RelationalData
    ) -> None:
        """
        No-op. Under this strategy, any collection of tables can be preserved.
        """
        pass

    def get_preserved_data(self, table: str, rel_data: RelationalData) -> pd.DataFrame:
        """
        Returns preserved source data for synthetic children
        to reference during generation post-processing.
        """
        return rel_data.get_table_data(table)

    def ready_to_generate(
        self,
        rel_data: RelationalData,
        in_progress: list[str],
        finished: list[str],
    ) -> list[str]:
        """
        All tables are immediately ready for generation. Once they are
        at least in progress, they are no longer ready.
        """
        return [
            table
            for table in rel_data.list_all_tables()
            if table not in in_progress and table not in finished
        ]

    def get_generation_job(
        self,
        table: str,
        rel_data: RelationalData,
        record_size_ratio: float,
        output_tables: dict[str, pd.DataFrame],
        subdir: str,
        output_handler: OutputHandler,
    ) -> dict[str, Any]:
        """
        Returns kwargs for a record handler job requesting an output record
        count based on the initial table data size and the record size ratio.
        """
        source_data_size = len(rel_data.get_table_data(table))
        synth_size = int(source_data_size * record_size_ratio)
        return {"params": {"num_records": synth_size}}

    def tables_to_skip_when_failed(
        self, table: str, rel_data: RelationalData
    ) -> list[str]:
        return []

    def post_process_individual_synthetic_result(
        self,
        table_name: str,
        rel_data: RelationalData,
        synthetic_table: pd.DataFrame,
        record_size_ratio: float,
    ) -> pd.DataFrame:
        """
        No-op. This strategy does not apply any changes to individual table results upon record handler completion.
        All post-processing is performed on the output tables collectively when they are all finished.
        """
        return synthetic_table

    def post_process_synthetic_results(
        self,
        synth_tables: dict[str, pd.DataFrame],
        preserved: list[str],
        rel_data: RelationalData,
        record_size_ratio: float,
    ) -> dict[str, pd.DataFrame]:
        "Synthesizes primary and foreign keys"
        synth_tables = _synthesize_primary_keys(
            synth_tables, preserved, rel_data, record_size_ratio
        )
        synth_tables = _synthesize_foreign_keys(synth_tables, rel_data)
        return synth_tables


def _synthesize_primary_keys(
    synth_tables: dict[str, pd.DataFrame],
    preserved: list[str],
    rel_data: RelationalData,
    record_size_ratio: float,
) -> dict[str, pd.DataFrame]:
    """
    Alters primary key columns on all tables *except* preserved.
    Assumes the primary key column is of type integer.
    """
    processed = {}
    for table_name, synth_data in synth_tables.items():
        processed[table_name] = synth_data.copy()
        if table_name in preserved:
            continue

        primary_key = rel_data.get_primary_key(table_name)
        synth_row_count = len(synth_data)

        if len(primary_key) == 0:
            continue
        elif len(primary_key) == 1:
            processed[table_name][primary_key[0]] = [i for i in range(synth_row_count)]
        else:
            synthetic_pk_columns = common.make_composite_pks(
                table_name=table_name,
                rel_data=rel_data,
                primary_key=primary_key,
                synth_row_count=synth_row_count,
            )

            # make_composite_pks may not have created as many unique keys as we have
            # synthetic rows, so we truncate the table to avoid inserting NaN PKs.
            processed[table_name] = pd.concat(
                [
                    processed[table_name].head(len(synthetic_pk_columns)),
                    pd.DataFrame.from_records(synthetic_pk_columns),
                ],
                axis=1,
            )

    return processed


def _synthesize_foreign_keys(
    synth_tables: dict[str, pd.DataFrame], rel_data: RelationalData
) -> dict[str, pd.DataFrame]:
    """
    Alters foreign key columns on all tables (*including* those flagged as not to
    be synthesized to ensure joining to a synthesized parent table continues to work)
    by replacing foreign key column values with valid values from the parent table column
    being referenced.
    """
    processed = {}
    for table_name in rel_data.list_tables_parents_before_children():
        out_df = synth_tables.get(table_name)
        if out_df is None:
            continue
        for foreign_key in rel_data.get_foreign_keys(table_name):
            # We pull the parent from `processed` instead of from `synth_tables` because "this" table
            # (`table_name` / `out_df`) may have a FK pointing to a parent column that _is itself_ a FK to some third table.
            # We want to ensure the synthetic values we're using to populate "this" table's FK column are
            # the final output values we've produced for its parent table.
            # We are synthesizing foreign keys in parent->child order, so the parent table
            # should(*) already exist in the processed dict with its final synthetic values...
            parent_synth_table = processed.get(foreign_key.parent_table_name)
            if parent_synth_table is None:
                # (*)...BUT the parent table generation job may have failed and therefore not be present in either `processed` or `synth_tables`.
                # The synthetic data for "this" table may still be useful, but we do not have valid/any synthetic
                # values from the parent to set in "this" table's foreign key column. Instead of introducing dangling
                # pointers, set the entire column to None.
                synth_parent_values = [None] * len(foreign_key.parent_columns)
            else:
                synth_parent_values = parent_synth_table[
                    foreign_key.parent_columns
                ].values.tolist()

            original_table_data = rel_data.get_table_data(table_name)
            fk_frequencies = common.get_frequencies(
                original_table_data, foreign_key.columns
            )

            new_fk_values = _collect_values(
                synth_parent_values, fk_frequencies, len(out_df)
            )

            out_df[foreign_key.columns] = new_fk_values

        processed[table_name] = out_df

    return processed


def _collect_values(
    values: list,
    frequencies: list[int],
    total: int,
) -> list:
    freqs = sorted(frequencies)

    # Loop through frequencies in ascending order,
    # adding "that many" of the next valid value
    # to the output collection
    v = 0
    f = 0
    new_values = []
    while len(new_values) < total:
        fk_value = values[v]

        for _ in range(freqs[f]):
            new_values.append(fk_value)

        v = _safe_inc(v, values)
        f = _safe_inc(f, freqs)

    # trim potential excess
    new_values = new_values[0:total]

    # shuffle for realism
    random.shuffle(new_values)

    return new_values


def _safe_inc(i: int, col: list) -> int:
    i = i + 1
    if i == len(col):
        i = 0
    return i
