import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from gretel_client.projects.models import Model
from pandas.api.types import is_string_dtype

import gretel_trainer.relational.ancestry as ancestry
import gretel_trainer.relational.strategies.common as common
from gretel_trainer.relational.core import (
    MultiTableException,
    RelationalData,
    TableEvaluation,
)

logger = logging.getLogger(__name__)


class AncestralStrategy:
    @property
    def name(self) -> str:
        return "ancestral"

    @property
    def default_model(self) -> str:
        return "amplify"

    @property
    def supported_models(self) -> List[str]:
        return ["amplify"]

    def label_encode_keys(
        self, rel_data: RelationalData, tables: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        return common.label_encode_keys(rel_data, tables)

    def prepare_training_data(
        self, rel_data: RelationalData
    ) -> Dict[str, pd.DataFrame]:
        """
        Returns tables with all ancestor fields added,
        minus any highly-unique categorical fields from ancestors.
        Primary keys are modified on two records to accommodate a
        sufficiently wide range of synthetic values during seeding.
        Corresponding foreign keys are also modified accordingly.
        """
        all_tables = rel_data.list_all_tables()
        tableset_with_altered_keys = {}
        training_data = {}

        # First, create a new table set identical to source data
        for table_name in all_tables:
            tableset_with_altered_keys[table_name] = rel_data.get_table_data(table_name).copy()

        # Translate all PKs to a contiguous list of integers
        tableset_with_altered_keys = common.label_encode_keys(rel_data, tableset_with_altered_keys)

        # On each table, add an artifical row with the max possible PK value
        max_pk_values = {}
        for table_name, data in tableset_with_altered_keys.items():
            pk = rel_data.get_primary_key(table_name)
            if pk is None:
                continue

            max_pk_values[table_name] = len(data) * 50

            last_record_copy = tableset_with_altered_keys[table_name].tail(1).copy()
            last_record_copy[pk] = max_pk_values[table_name]
            tableset_with_altered_keys[table_name] = pd.concat([data, last_record_copy]).reset_index(drop=True)

        # On each table with foreign keys, add two more artificial rows containing the min and max FK values
        for table_name, data in tableset_with_altered_keys.items():
            foreign_keys = rel_data.get_foreign_keys(table_name)
            if len(foreign_keys) == 0:
                continue

            pk = rel_data.get_primary_key(table_name)

            artificial_record = tableset_with_altered_keys[table_name].tail(1)
            min_fk_record = artificial_record.copy()
            max_fk_record = artificial_record.copy()

            for foreign_key in foreign_keys:
                min_fk_record[foreign_key.column_name] = 0
                max_fk_record[foreign_key.column_name] = max_pk_values[foreign_key.parent_table_name]

                if pk is not None:
                    min_fk_record[pk] = max_pk_values[table_name] + 1
                    max_fk_record[pk] = max_pk_values[table_name] + 2

            tableset_with_altered_keys[table_name] = pd.concat([data, min_fk_record, max_fk_record]).reset_index(drop=True)

        # Next, collect all data in multigenerational format
        for table_name in all_tables:
            data = ancestry.get_table_data_with_ancestors(
                rel_data, table_name, tableset_with_altered_keys
            )
            training_data[table_name] = data

        # Finally, drop highly-unique categorical ancestor fields
        for table_name, data in training_data.items():
            columns_to_drop = []

            for column in data.columns:
                if ancestry.is_ancestral_column(
                    column
                ) and _is_highly_unique_categorical(column, data):
                    columns_to_drop.append(column)
            training_data[table_name] = data.drop(columns=columns_to_drop)

        return training_data

    def tables_to_retrain(
        self, tables: List[str], rel_data: RelationalData
    ) -> List[str]:
        """
        Given a set of tables requested to retrain, returns those tables with all their
        descendants, because those descendant tables were trained with data from their
        parents appended.
        """
        retrain = set(tables)
        for table in tables:
            retrain.update(rel_data.get_descendants(table))
        return list(retrain)

    def validate_preserved_tables(
        self, tables: List[str], rel_data: RelationalData
    ) -> None:
        """
        Ensures that for every table marked as preserved, all its ancestors are also preserved.
        """
        for table in tables:
            for parent in rel_data.get_parents(table):
                if parent not in tables:
                    raise MultiTableException(
                        f"Cannot preserve table {table} without also preserving parent {parent}."
                    )

    def get_preserved_data(self, table: str, rel_data: RelationalData) -> pd.DataFrame:
        """
        Returns preserved source data in multigenerational format for synthetic children
        to reference during generation post-processing.
        """
        return ancestry.get_table_data_with_ancestors(rel_data, table)

    def ready_to_generate(
        self,
        rel_data: RelationalData,
        in_progress: List[str],
        finished: List[str],
    ) -> List[str]:
        """
        Tables with no parents are immediately ready for generation.
        Tables with parents are ready once their parents are finished.
        All tables are no longer considered ready once they are at least in progress.
        """
        ready = []

        for table in rel_data.list_all_tables():
            if table in in_progress or table in finished:
                continue

            parents = rel_data.get_parents(table)
            if len(parents) == 0:
                ready.append(table)
            elif all([parent in finished for parent in parents]):
                ready.append(table)

        return ready

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
        Returns kwargs for creating a record handler job via the Gretel SDK.

        If the table does not have any parents, job requests an output
        record count based on the initial table data size and the record size ratio.

        If the table does have parents, builds a seed dataframe to use in generation.
        """
        source_data_size = len(rel_data.get_table_data(table))
        synth_size = int(source_data_size * record_size_ratio)
        if len(rel_data.get_parents(table)) == 0:
            return {"params": {"num_records": synth_size}}
        else:
            seed_df = self._build_seed_data_for_table(
                table, output_tables, rel_data, synth_size, training_columns
            )
            seed_path = working_dir / f"synthetics_seed_{table}.csv"
            seed_df.to_csv(seed_path, index=False)
            return {"data_source": str(seed_path)}

    def _build_seed_data_for_table(
        self,
        table: str,
        output_tables: Dict[str, pd.DataFrame],
        rel_data: RelationalData,
        synth_size: int,
        training_columns: List[str],
    ) -> pd.DataFrame:
        seed_df = pd.DataFrame()

        for fk in rel_data.get_foreign_keys(table):
            parent_table_data = output_tables[fk.parent_table_name]
            parent_table_data = ancestry.prepend_foreign_key_lineage(
                parent_table_data, fk.column_name
            )

            # Get FK frequencies
            freqs = (
                rel_data.get_table_data(table)
                .groupby([fk.column_name])
                .size()
                .reset_index()
            )
            freqs = sorted(list(freqs[0]), reverse=True)
            f = 0

            # Make a list of parent_table indicies matching FK frequencies
            parent_indices = range(len(parent_table_data))
            p = 0
            parent_indices_to_use_as_fks = []
            while len(parent_indices_to_use_as_fks) < synth_size:
                parent_index_to_use = parent_indices[p]
                for _ in range(freqs[f]):
                    parent_indices_to_use_as_fks.append(parent_index_to_use)
                p = _safe_inc(p, parent_indices)
                f = _safe_inc(f, freqs)

            # Turn list into a DF and merge the parent table data
            tmp_column_name = "tmp_parent_merge"
            this_fk_seed_df = pd.DataFrame(
                data={tmp_column_name: parent_indices_to_use_as_fks}
            )
            this_fk_seed_df = this_fk_seed_df.merge(
                parent_table_data,
                how="left",
                left_on=tmp_column_name,
                right_index=True,
            )

            # Drop any columns that weren't used in training, as well as the temporary merge column
            columns_to_drop = [
                col for col in this_fk_seed_df.columns if col not in training_columns
            ]
            columns_to_drop.append(tmp_column_name)
            this_fk_seed_df = this_fk_seed_df.drop(columns=columns_to_drop)

            seed_df = pd.concat(
                [
                    seed_df.reset_index(drop=True),
                    this_fk_seed_df.reset_index(drop=True),
                ],
                axis=1,
            )

        return seed_df

    def post_process_individual_synthetic_result(
        self,
        table_name: str,
        rel_data: RelationalData,
        synthetic_table: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Replaces primary key values with a new, contiguous set of values.
        Replaces synthesized foreign keys with seed primary keys.
        """
        primary_key = ancestry.get_multigenerational_primary_key(rel_data, table_name)
        if primary_key is None:
            return synthetic_table

        processed = synthetic_table
        processed[primary_key] = [i for i in range(len(synthetic_table))]

        foreign_key_maps = ancestry.get_ancestral_foreign_key_maps(rel_data, table_name)
        for fk, parent_pk in foreign_key_maps:
            processed[fk] = processed[parent_pk]

        return processed

    def post_process_synthetic_results(
        self,
        output_tables: Dict[str, pd.DataFrame],
        preserved: List[str],
        rel_data: RelationalData,
    ) -> Dict[str, pd.DataFrame]:
        """
        Restores tables from multigenerational to original shape
        """
        # TODO: do we need to do any additional PK/FK manipulation?
        return {
            table_name: ancestry.drop_ancestral_data(df)
            for table_name, df in output_tables.items()
        }

    def update_evaluation_from_model(
        self,
        table_name: str,
        evaluations: Dict[str, TableEvaluation],
        model: Model,
        working_dir: Path,
    ) -> None:
        logger.info(f"Downloading cross_table evaluation reports for `{table_name}`.")
        out_filepath = working_dir / f"synthetics_cross_table_evaluation_{table_name}"
        common.download_artifacts(model, out_filepath, table_name)

        evaluation = evaluations[table_name]
        evaluation.cross_table_sqs = common.get_sqs_score(model)
        evaluation.cross_table_report_json = common.read_report_json_data(
            model, out_filepath
        )

    def update_evaluation_via_evaluate(
        self,
        evaluation: TableEvaluation,
        table: str,
        rel_data: RelationalData,
        synthetic_tables: Dict[str, pd.DataFrame],
        working_dir: Path,
    ) -> None:
        source_data = rel_data.get_table_data(table)
        synth_data = synthetic_tables[table]

        logger.info(f"Running individual evaluations for `{table}`.")
        report = common.get_quality_report(
            source_data=source_data, synth_data=synth_data
        )
        out_filepath = working_dir / f"synthetics_individual_evaluation_{table}"
        common.write_report(report, out_filepath)

        evaluation.individual_sqs = report.peek().get("score")
        evaluation.individual_report_json = report.as_dict


def _is_highly_unique_categorical(col: str, df: pd.DataFrame) -> bool:
    return is_string_dtype(df[col]) and _percent_unique(col, df) >= 0.7


def _percent_unique(col: str, df: pd.DataFrame) -> float:
    col_no_nan = df[col].dropna()
    total = len(col_no_nan)
    distinct = col_no_nan.nunique()

    if total == 0:
        return 0.0
    else:
        return distinct / total


def _safe_inc(i: int, col: Union[List[Any], range]) -> int:
    i = i + 1
    if i == len(col):
        i = 0
    return i
