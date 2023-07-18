import logging
import random
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from gretel_client.projects.models import Model

import gretel_trainer.relational.ancestry as ancestry
import gretel_trainer.relational.strategies.common as common
from gretel_trainer.relational.core import RelationalData
from gretel_trainer.relational.sdk_extras import ExtendedGretelSDK
from gretel_trainer.relational.table_evaluation import TableEvaluation

logger = logging.getLogger(__name__)


class IndependentStrategy:
    @property
    def name(self) -> str:
        return "independent"

    # TODO: remove when `gretel_model` param is removed
    @property
    def supported_gretel_models(self) -> list[str]:
        return ["amplify", "actgan", "lstm", "tabular-dp"]

    @property
    def supported_model_keys(self) -> list[str]:
        return ["amplify", "actgan", "synthetics", "tabular_dp"]

    def label_encode_keys(
        self, rel_data: RelationalData, tables: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        return common.label_encode_keys(rel_data, tables)

    def prepare_training_data(
        self, rel_data: RelationalData, table_paths: dict[str, Path]
    ) -> dict[str, Path]:
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

            pd.DataFrame(columns=use_columns).to_csv(path, index=False)
            source_path = rel_data.get_table_source(table)
            for chunk in pd.read_csv(
                source_path, usecols=use_columns, chunksize=10_000
            ):
                chunk.to_csv(path, index=False, mode="a", header=False)

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
        target_dir: Path,
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

    def update_evaluation_from_model(
        self,
        table_name: str,
        evaluations: dict[str, TableEvaluation],
        model: Model,
        working_dir: Path,
        extended_sdk: ExtendedGretelSDK,
    ) -> None:
        logger.info(f"Downloading individual evaluation reports for `{table_name}`.")
        out_filepath = working_dir / f"synthetics_individual_evaluation_{table_name}"
        common.download_artifacts(model, out_filepath, extended_sdk)

        evaluation = evaluations[table_name]
        evaluation.individual_report_json = common.read_report_json_data(
            model, out_filepath
        )

    def get_evaluate_model_data(
        self,
        table_name: str,
        rel_data: RelationalData,
        synthetic_tables: dict[str, pd.DataFrame],
    ) -> Optional[dict[str, pd.DataFrame]]:
        missing_ancestors = [
            ancestor
            for ancestor in rel_data.get_ancestors(table_name)
            if ancestor not in synthetic_tables
        ]
        if len(missing_ancestors) > 0:
            logger.info(
                f"Cannot run cross_table evaluations for `{table_name}` because no synthetic data exists for ancestor tables {missing_ancestors}."
            )
            return None

        source_data = ancestry.get_table_data_with_ancestors(rel_data, table_name)
        synthetic_data = ancestry.get_table_data_with_ancestors(
            rel_data, table_name, synthetic_tables
        )
        return {
            "source": source_data,
            "synthetic": synthetic_data,
        }

    def update_evaluation_from_evaluate(
        self,
        table_name: str,
        evaluations: dict[str, TableEvaluation],
        evaluate_model: Model,
        working_dir: Path,
        extended_sdk: ExtendedGretelSDK,
    ) -> None:
        logger.info(f"Downloading cross table evaluation reports for `{table_name}`.")
        out_filepath = working_dir / f"synthetics_cross_table_evaluation_{table_name}"
        common.download_artifacts(evaluate_model, out_filepath, extended_sdk)

        evaluation = evaluations[table_name]
        evaluation.cross_table_report_json = common.read_report_json_data(
            evaluate_model, out_filepath
        )


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
    for table_name, synth_data in synth_tables.items():
        out_df = synth_data.copy()
        for foreign_key in rel_data.get_foreign_keys(table_name):
            parent_synth_table = synth_tables.get(foreign_key.parent_table_name)
            if parent_synth_table is None:
                # Parent table generation job may have failed and therefore not be present in synth_tables.
                # The synthetic data for this table may still be useful, but we do not have valid synthetic
                # primary key values to set in this table's foreign key column. Instead of introducing dangling
                # pointers, set the entire column to None.
                synth_pk_values = [None] * len(foreign_key.parent_columns)
            else:
                synth_pk_values = parent_synth_table[
                    foreign_key.parent_columns
                ].values.tolist()

            original_table_data = rel_data.get_table_data(table_name)
            fk_frequencies = common.get_frequencies(
                original_table_data, foreign_key.columns
            )

            new_fk_values = _collect_values(
                synth_pk_values, fk_frequencies, len(out_df)
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
