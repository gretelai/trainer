import logging
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
from gretel_client.projects.models import Model

import gretel_trainer.relational.ancestry as ancestry
import gretel_trainer.relational.strategies.common as common
from gretel_trainer.relational.core import MultiTableException, RelationalData
from gretel_trainer.relational.sdk_extras import ExtendedGretelSDK
from gretel_trainer.relational.table_evaluation import TableEvaluation

logger = logging.getLogger(__name__)


class AncestralStrategy:
    @property
    def name(self) -> str:
        return "ancestral"

    # TODO: remove when `gretel_model` param is removed
    @property
    def supported_gretel_models(self) -> list[str]:
        return ["amplify"]

    @property
    def supported_model_keys(self) -> list[str]:
        return ["amplify"]

    def label_encode_keys(
        self, rel_data: RelationalData, tables: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        return common.label_encode_keys(rel_data, tables)

    def prepare_training_data(
        self, rel_data: RelationalData, table_paths: dict[str, Path]
    ) -> dict[str, Path]:
        """
        Writes tables' training data to provided paths.
        Training data has:
        - all safe-for-seed ancestor fields added
        - columns in multigenerational format
        - all keys translated to contiguous integers
        - artificial min/max seed records added
        """
        all_tables = rel_data.list_all_tables()
        omitted_tables = [t for t in all_tables if t not in table_paths]
        altered_tableset = {}

        # Create a new table set identical to source data
        for table_name in all_tables:
            altered_tableset[table_name] = rel_data.get_table_data(table_name).copy()

        # Translate all keys to a contiguous list of integers
        altered_tableset = common.label_encode_keys(
            rel_data, altered_tableset, omit=omitted_tables
        )

        # Add artificial rows to support seeding
        altered_tableset = _add_artifical_rows_for_seeding(
            rel_data, altered_tableset, omitted_tables
        )

        # Collect all data in multigenerational format
        for table, path in table_paths.items():
            data = ancestry.get_table_data_with_ancestors(
                rel_data=rel_data,
                table=table,
                tableset=altered_tableset,
                ancestral_seeding=True,
            )
            data.to_csv(path, index=False)

        return table_paths

    def tables_to_retrain(
        self, tables: list[str], rel_data: RelationalData
    ) -> list[str]:
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
        self, tables: list[str], rel_data: RelationalData
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
        in_progress: list[str],
        finished: list[str],
    ) -> list[str]:
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
        output_tables: dict[str, pd.DataFrame],
        target_dir: Path,
    ) -> dict[str, Any]:
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
                table, output_tables, rel_data, synth_size
            )
            seed_path = target_dir / f"synthetics_seed_{table}.csv"
            seed_df.to_csv(seed_path, index=False)
            return {"data_source": str(seed_path)}

    def _build_seed_data_for_table(
        self,
        table: str,
        output_tables: dict[str, pd.DataFrame],
        rel_data: RelationalData,
        synth_size: int,
    ) -> pd.DataFrame:
        column_legend = ancestry.get_seed_safe_multigenerational_columns(rel_data)
        seed_df = pd.DataFrame()

        source_data = rel_data.get_table_data(table)
        for fk in rel_data.get_foreign_keys(table):
            parent_table_data = output_tables[fk.parent_table_name]
            parent_table_data = ancestry.prepend_foreign_key_lineage(
                parent_table_data, fk.columns
            )

            # Get FK frequencies
            freqs = common.get_frequencies(source_data, fk.columns)
            freqs = sorted(freqs, reverse=True)
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
                col
                for col in this_fk_seed_df.columns
                if col not in column_legend[table]
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
    ) -> list[str]:
        return rel_data.get_descendants(table)

    def post_process_individual_synthetic_result(
        self,
        table_name: str,
        rel_data: RelationalData,
        synthetic_table: pd.DataFrame,
        record_size_ratio: float,
    ) -> pd.DataFrame:
        """
        Replaces primary key values with a new, contiguous set of values.
        Replaces synthesized foreign keys with seed primary keys.
        """
        processed = synthetic_table

        multigenerational_primary_key = ancestry.get_multigenerational_primary_key(
            rel_data, table_name
        )

        if len(multigenerational_primary_key) == 0:
            pass
        elif len(multigenerational_primary_key) == 1:
            processed[multigenerational_primary_key[0]] = [
                i for i in range(len(synthetic_table))
            ]
        else:
            synthetic_pk_columns = common.make_composite_pks(
                table_name=table_name,
                rel_data=rel_data,
                primary_key=multigenerational_primary_key,
                synth_row_count=len(synthetic_table),
            )

            # make_composite_pks may not have created as many unique keys as we have
            # synthetic rows, so we truncate the table to avoid inserting NaN PKs.
            processed = pd.concat(
                [
                    pd.DataFrame.from_records(synthetic_pk_columns),
                    processed.drop(multigenerational_primary_key, axis="columns").head(
                        len(synthetic_pk_columns)
                    ),
                ],
                axis=1,
            )

        for fk_map in ancestry.get_ancestral_foreign_key_maps(rel_data, table_name):
            fk_col, parent_pk_col = fk_map
            processed[fk_col] = processed[parent_pk_col]

        return processed

    def post_process_synthetic_results(
        self,
        synth_tables: dict[str, pd.DataFrame],
        preserved: list[str],
        rel_data: RelationalData,
        record_size_ratio: float,
    ) -> dict[str, pd.DataFrame]:
        """
        Restores tables from multigenerational to original shape
        """
        return {
            table_name: ancestry.drop_ancestral_data(df)
            for table_name, df in synth_tables.items()
        }

    def update_evaluation_from_model(
        self,
        table_name: str,
        evaluations: dict[str, TableEvaluation],
        model: Model,
        working_dir: Path,
        extended_sdk: ExtendedGretelSDK,
    ) -> None:
        logger.info(f"Downloading cross_table evaluation reports for `{table_name}`.")
        out_filepath = working_dir / f"synthetics_cross_table_evaluation_{table_name}"
        common.download_artifacts(model, out_filepath, extended_sdk)

        evaluation = evaluations[table_name]
        evaluation.cross_table_report_json = common.read_report_json_data(
            model, out_filepath
        )

    def get_evaluate_model_data(
        self,
        table_name: str,
        rel_data: RelationalData,
        synthetic_tables: dict[str, pd.DataFrame],
    ) -> Optional[dict[str, pd.DataFrame]]:
        return {
            "source": rel_data.get_table_data(table_name),
            "synthetic": synthetic_tables[table_name],
        }

    def update_evaluation_from_evaluate(
        self,
        table_name: str,
        evaluations: dict[str, TableEvaluation],
        evaluate_model: Model,
        working_dir: Path,
        extended_sdk: ExtendedGretelSDK,
    ) -> None:
        logger.info(f"Downloading individual evaluation reports for `{table_name}`.")
        out_filepath = working_dir / f"synthetics_individual_evaluation_{table_name}"
        common.download_artifacts(evaluate_model, out_filepath, extended_sdk)

        evaluation = evaluations[table_name]
        evaluation.individual_report_json = common.read_report_json_data(
            evaluate_model, out_filepath
        )


def _add_artifical_rows_for_seeding(
    rel_data: RelationalData, tables: dict[str, pd.DataFrame], omitted: list[str]
) -> dict[str, pd.DataFrame]:
    # On each table, add an artifical row with the max possible PK value
    # unless the table is omitted from synthetics
    max_pk_values = {}
    for table_name, data in tables.items():
        if table_name in omitted:
            continue
        max_pk_values[table_name] = len(data) * 50

        random_record = tables[table_name].sample().copy()
        for pk_col in rel_data.get_primary_key(table_name):
            random_record[pk_col] = max_pk_values[table_name]
        tables[table_name] = pd.concat([data, random_record]).reset_index(drop=True)

    # On each table with foreign keys, add two more artificial rows containing the min and max FK values
    for table_name, data in tables.items():
        foreign_keys = rel_data.get_foreign_keys(table_name)
        if len(foreign_keys) == 0:
            continue

        # Skip if the parent table is omitted and is the only parent
        if len(foreign_keys) == 1 and foreign_keys[0].parent_table_name in omitted:
            continue

        two_records = tables[table_name].sample(2)
        min_fk_record = two_records.head(1).copy()
        max_fk_record = two_records.tail(1).copy()

        # By default, just auto-increment the primary key
        for pk_col in rel_data.get_primary_key(table_name):
            min_fk_record[pk_col] = max_pk_values[table_name] + 1
            max_fk_record[pk_col] = max_pk_values[table_name] + 2

        # This can potentially overwrite the auto-incremented primary keys above in the case of composite keys
        for foreign_key in foreign_keys:
            # Treat FK columns to omitted parents as normal columns
            if foreign_key.parent_table_name in omitted:
                continue
            for fk_col in foreign_key.columns:
                min_fk_record[fk_col] = 0
                max_fk_record[fk_col] = max_pk_values[foreign_key.parent_table_name]

        tables[table_name] = pd.concat(
            [data, min_fk_record, max_fk_record]
        ).reset_index(drop=True)

    return tables


def _safe_inc(i: int, col: Union[list, range]) -> int:
    i = i + 1
    if i == len(col):
        i = 0
    return i
