import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

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

    @property
    def default_model(self) -> str:
        return "amplify"

    @property
    def supported_models(self) -> List[str]:
        return ["amplify", "actgan", "lstm", "tabular-dp"]

    def label_encode_keys(
        self, rel_data: RelationalData, tables: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        return common.label_encode_keys(rel_data, tables)

    def prepare_training_data(
        self, rel_data: RelationalData
    ) -> Dict[str, pd.DataFrame]:
        """
        Returns source tables with primary and foreign keys removed
        """
        training_data = {}

        for table_name in rel_data.list_all_tables():
            columns_to_drop = []
            columns_to_drop.extend(rel_data.get_primary_key(table_name))
            for foreign_key in rel_data.get_foreign_keys(table_name):
                columns_to_drop.extend(foreign_key.columns)

            data = rel_data.get_table_data(table_name)
            data = data.drop(columns=columns_to_drop)

            training_data[table_name] = data

        return training_data

    def tables_to_retrain(
        self, tables: List[str], rel_data: RelationalData
    ) -> List[str]:
        """
        Returns the provided tables requested to retrain, unaltered.
        """
        return tables

    def validate_preserved_tables(
        self, tables: List[str], rel_data: RelationalData
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
        in_progress: List[str],
        finished: List[str],
    ) -> List[str]:
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
        output_tables: Dict[str, pd.DataFrame],
        target_dir: Path,
        training_columns: List[str],
    ) -> Dict[str, Any]:
        """
        Returns kwargs for a record handler job requesting an output record
        count based on the initial table data size and the record size ratio.
        """
        source_data_size = len(rel_data.get_table_data(table))
        synth_size = int(source_data_size * record_size_ratio)
        return {"params": {"num_records": synth_size}}

    def tables_to_skip_when_failed(
        self, table: str, rel_data: RelationalData
    ) -> List[str]:
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
        synth_tables: Dict[str, pd.DataFrame],
        preserved: List[str],
        rel_data: RelationalData,
        record_size_ratio: float,
    ) -> Dict[str, pd.DataFrame]:
        "Synthesizes primary and foreign keys"
        synth_tables = _synthesize_primary_keys(
            synth_tables, preserved, rel_data, record_size_ratio
        )
        synth_tables = _synthesize_foreign_keys(synth_tables, rel_data)
        return synth_tables

    def update_evaluation_from_model(
        self,
        table_name: str,
        evaluations: Dict[str, TableEvaluation],
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
        synthetic_tables: Dict[str, pd.DataFrame],
    ) -> Optional[Dict[str, pd.DataFrame]]:
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
        evaluations: Dict[str, TableEvaluation],
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
    synth_tables: Dict[str, pd.DataFrame],
    preserved: List[str],
    rel_data: RelationalData,
    record_size_ratio: float,
) -> Dict[str, pd.DataFrame]:
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
            synthetic_pk_columns = common.make_composite_pk_columns(
                table_name=table_name,
                rel_data=rel_data,
                primary_key=primary_key,
                synth_row_count=synth_row_count,
                record_size_ratio=record_size_ratio,
            )
            for index, col in enumerate(primary_key):
                processed[table_name][col] = synthetic_pk_columns[index]

    return processed


def _synthesize_foreign_keys(
    synth_tables: Dict[str, pd.DataFrame], rel_data: RelationalData
) -> Dict[str, pd.DataFrame]:
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
    values: List[Any],
    frequencies: List[int],
    total: int,
) -> List[Any]:
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


def _safe_inc(i: int, col: List[Any]) -> int:
    i = i + 1
    if i == len(col):
        i = 0
    return i
