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
        Returns tables with:
        - all ancestor fields added
        - columns in multigenerational format
        - all keys translated to contiguous integers
        - artificial min/max seed records added
        - known-problematic fields removed
        """
        all_tables = rel_data.list_all_tables()
        altered_tableset = {}
        training_data = {}

        # Create a new table set identical to source data
        for table_name in all_tables:
            altered_tableset[table_name] = rel_data.get_table_data(table_name).copy()

        # Translate all keys to a contiguous list of integers
        altered_tableset = common.label_encode_keys(rel_data, altered_tableset)

        # Add artificial rows to support seeding
        altered_tableset = _add_artifical_rows_for_seeding(rel_data, altered_tableset)

        # Collect all data in multigenerational format
        for table_name in all_tables:
            data = ancestry.get_table_data_with_ancestors(
                rel_data, table_name, altered_tableset
            )
            training_data[table_name] = data

        # Drop some columns known to be problematic
        for table_name, data in training_data.items():
            columns_to_drop = [
                col for col in data.columns if _drop_from_training(col, data)
            ]
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
        target_dir: Path,
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
            seed_path = target_dir / f"synthetics_seed_{table}.csv"
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

    def tables_to_skip_when_failed(
        self, table: str, rel_data: RelationalData
    ) -> List[str]:
        return rel_data.get_descendants(table)

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
        processed = synthetic_table

        primary_key = ancestry.get_multigenerational_primary_key(rel_data, table_name)
        if primary_key is not None:
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
        evaluation.cross_table_report_json = common.read_report_json_data(
            model, out_filepath
        )

    def get_evaluate_model_data(
        self,
        table_name: str,
        rel_data: RelationalData,
        synthetic_tables: Dict[str, pd.DataFrame],
    ) -> Optional[Dict[str, pd.DataFrame]]:
        return {
            "source": rel_data.get_table_data(table_name),
            "synthetic": synthetic_tables[table_name],
        }

    def update_evaluation_from_evaluate(
        self,
        table_name: str,
        evaluations: Dict[str, TableEvaluation],
        evaluate_model: Model,
        working_dir: Path,
    ) -> None:
        logger.info(f"Downloading individual evaluation reports for `{table_name}`.")
        out_filepath = working_dir / f"synthetics_individual_evaluation_{table_name}"
        common.download_artifacts(evaluate_model, out_filepath, table_name)

        evaluation = evaluations[table_name]
        evaluation.individual_report_json = common.read_report_json_data(
            evaluate_model, out_filepath
        )


def _add_artifical_rows_for_seeding(
    rel_data: RelationalData, tables: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    # On each table, add an artifical row with the max possible PK value
    max_pk_values = {}
    for table_name, data in tables.items():
        pk = rel_data.get_primary_key(table_name)
        if pk is None:
            continue

        max_pk_values[table_name] = len(data) * 50

        random_record = tables[table_name].sample().copy()
        random_record[pk] = max_pk_values[table_name]
        tables[table_name] = pd.concat([data, random_record]).reset_index(drop=True)

    # On each table with foreign keys, add two more artificial rows containing the min and max FK values
    for table_name, data in tables.items():
        foreign_keys = rel_data.get_foreign_keys(table_name)
        if len(foreign_keys) == 0:
            continue

        pk = rel_data.get_primary_key(table_name)

        two_records = tables[table_name].sample(2)
        min_fk_record = two_records.head(1).copy()
        max_fk_record = two_records.tail(1).copy()

        for foreign_key in foreign_keys:
            min_fk_record[foreign_key.column_name] = 0
            max_fk_record[foreign_key.column_name] = max_pk_values[
                foreign_key.parent_table_name
            ]

        if pk is not None:
            min_fk_record[pk] = max_pk_values[table_name] + 1
            max_fk_record[pk] = max_pk_values[table_name] + 2

        tables[table_name] = pd.concat(
            [data, min_fk_record, max_fk_record]
        ).reset_index(drop=True)

    return tables


def _drop_from_training(col: str, df: pd.DataFrame) -> bool:
    return ancestry.is_ancestral_column(col) and (
        _is_highly_unique_categorical(col, df) or _is_highly_nan(col, df)
    )


def _is_highly_nan(col: str, df: pd.DataFrame) -> bool:
    missing = df[col].isnull().sum()
    missing_perc = missing / len(df)
    return missing_perc > 0.2


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
