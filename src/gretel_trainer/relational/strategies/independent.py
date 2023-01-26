import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from gretel_client.projects.models import Model

import gretel_trainer.relational.ancestry as ancestry
import gretel_trainer.relational.strategies.common as common
from gretel_trainer.relational.core import RelationalData, TableEvaluation


class IndependentStrategy:
    @property
    def default_model(self) -> str:
        return "amplify"

    @property
    def supported_models(self) -> List[str]:
        return ["amplify", "actgan", "lstm"]

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
            data = rel_data.get_table_data(table_name)
            columns_to_drop = []

            primary_key = rel_data.get_primary_key(table_name)
            if primary_key is not None:
                columns_to_drop.append(primary_key)
            foreign_keys = rel_data.get_foreign_keys(table_name)
            columns_to_drop.extend(
                [foreign_key.column_name for foreign_key in foreign_keys]
            )
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
        working_dir: Path,
        training_columns: List[str],
    ) -> Dict[str, Any]:
        """
        Returns kwargs for a record handler job requesting an output record
        count based on the initial table data size and the record size ratio.
        """
        source_data_size = len(rel_data.get_table_data(table))
        synth_size = int(source_data_size * record_size_ratio)
        return {"params": {"num_records": synth_size}}

    def post_process_individual_synthetic_result(
        self,
        table_name: str,
        rel_data: RelationalData,
        synthetic_table: pd.DataFrame,
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
    ) -> Dict[str, pd.DataFrame]:
        "Synthesizes primary and foreign keys"
        synth_tables = _synthesize_primary_keys(synth_tables, preserved, rel_data)
        synth_tables = _synthesize_foreign_keys(synth_tables, rel_data)
        return synth_tables

    def update_evaluation_from_model(
        self,
        table_name: str,
        evaluations: Dict[str, TableEvaluation],
        model: Model,
        working_dir: Path,
    ) -> None:
        artifacts_dir = common.download_artifacts(model, table_name, working_dir)

        evaluation = evaluations[table_name]
        evaluation.individual_sqs = common.get_sqs_score(model)
        evaluation.individual_report_json = common.read_report_json_data(
            model, artifacts_dir
        )

    def update_evaluation_via_evaluate(
        self,
        evaluation: TableEvaluation,
        table: str,
        rel_data: RelationalData,
        synthetic_tables: Dict[str, pd.DataFrame],
        working_dir: Path,
    ) -> None:
        source_data = ancestry.get_table_data_with_ancestors(rel_data, table)
        synth_data = ancestry.get_table_data_with_ancestors(
            rel_data, table, synthetic_tables
        )

        report = common.get_quality_report(
            source_data=source_data, synth_data=synth_data
        )
        common.write_report(report, table, working_dir)

        evaluation.cross_table_sqs = report.peek().get("score")
        evaluation.cross_table_report_json = report.as_dict


def _synthesize_primary_keys(
    synth_tables: Dict[str, pd.DataFrame],
    preserved: List[str],
    rel_data: RelationalData,
) -> Dict[str, pd.DataFrame]:
    """
    Alters primary key columns on all tables *except* preserved.
    Assumes the primary key column is of type integer.
    """
    processed = {}
    for table_name, synth_data in synth_tables.items():
        out_df = synth_data.copy()
        if table_name in preserved:
            processed[table_name] = out_df
            continue

        primary_key = rel_data.get_primary_key(table_name)
        if primary_key is None:
            processed[table_name] = out_df
            continue

        out_df[primary_key] = [i for i in range(len(synth_data))]
        processed[table_name] = out_df

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
            parent_synth_table = synth_tables[foreign_key.parent_table_name]
            synth_pk_values = list(parent_synth_table[foreign_key.parent_column_name])

            original_table_data = rel_data.get_table_data(table_name)
            original_fk_frequencies = (
                original_table_data.groupby(foreign_key.column_name)
                .size()
                .reset_index()
            )
            frequencies = list(original_fk_frequencies[0])

            new_fk_values = _collect_new_foreign_key_values(
                synth_pk_values, frequencies, len(out_df)
            )

            out_df[foreign_key.column_name] = new_fk_values

        processed[table_name] = out_df

    return processed


def _collect_new_foreign_key_values(
    values: List[Any],
    frequencies: List[int],
    total: int,
) -> List[Any]:
    freqs = sorted(frequencies)

    # Loop through frequencies in ascending order,
    # adding "that many" of the next valid FK value
    # to the output collection
    v = 0
    f = 0
    new_fk_values = []
    while len(new_fk_values) < total:
        fk_value = values[v]

        for _ in range(freqs[f]):
            new_fk_values.append(fk_value)

        v = _safe_inc(v, values)
        f = _safe_inc(f, freqs)

    # trim potential excess
    new_fk_values = new_fk_values[0:total]

    # shuffle for realism
    random.shuffle(new_fk_values)

    return new_fk_values


def _safe_inc(i: int, col: List[Any]) -> int:
    i = i + 1
    if i == len(col):
        i = 0
    return i
