import itertools
from typing import Any, Dict, List, Optional

import pandas as pd
from gretel_client.projects.models import Model
from pandas.api.types import is_string_dtype

import gretel_trainer.relational.strategies.common as common
from gretel_trainer.relational.core import (
    MultiTableException,
    RelationalData,
    TableEvaluation,
)


class CrossTableStrategy:
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
            tableset_with_altered_keys[table_name] = rel_data.get_table_data(table_name)

        # On each table, alter the PKs in the first two rows for the
        # min/max seed range, plus alter all FK references to those records
        for table_name, data in tableset_with_altered_keys.items():
            pk = rel_data.get_primary_key(table_name)
            if pk is None:
                continue

            orig_pk_min = data.loc[0][pk]
            orig_pk_max = data.loc[1][pk]

            new_pk_min = 0
            new_pk_max = len(data) * 50

            # Mutate pk values on table
            data.loc[0, [pk]] = [new_pk_min]
            data.loc[1, [pk]] = [new_pk_max]

            # Update FKs to match
            for other_table_name in all_tables:
                if other_table_name == table_name:
                    continue
                fks = rel_data.get_foreign_keys(other_table_name)
                for fk in fks:
                    if (
                        fk.parent_table_name == table_name
                        and fk.parent_column_name == pk
                    ):
                        other_table_data = tableset_with_altered_keys[other_table_name]
                        modified = other_table_data.replace(
                            {
                                fk.column_name: {
                                    orig_pk_min: new_pk_min,
                                    orig_pk_max: new_pk_max,
                                }
                            }
                        )
                        tableset_with_altered_keys[other_table_name] = modified

        # Next, collect all data in multigenerational format
        for table_name in all_tables:
            data = rel_data.get_table_data_with_ancestors(
                table_name, tableset_with_altered_keys
            )
            training_data[table_name] = data

        # Finally, drop highly-unique categorical ancestor fields
        for table_name, data in training_data.items():
            columns_to_drop = []

            for column in data.columns:
                if rel_data.is_ancestral_column(
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
                table, output_tables, rel_data, synth_size
            )
            return {"data_source": seed_df}

    def _build_seed_data_for_table(
        self,
        table: str,
        output_tables: Dict[str, pd.DataFrame],
        rel_data: RelationalData,
        synth_size: int,
    ) -> pd.DataFrame:
        seed_df = pd.DataFrame()

        for fk in rel_data.get_foreign_keys(table):
            this_fk_seed_df = pd.DataFrame()

            parent_table_data = output_tables[fk.parent_table_name]
            parent_table_data = rel_data.prepend_foreign_key_lineage(
                parent_table_data, fk.column_name
            )
            parent_index_cycle = itertools.cycle(range(len(parent_table_data)))

            freqs = (
                rel_data.get_table_data(table)
                .groupby([fk.column_name])
                .size()
                .reset_index()
            )
            freqs = sorted(list(freqs[0]), reverse=True)
            freqs_cycle = itertools.cycle(freqs)

            while len(this_fk_seed_df) < synth_size:
                parent_record = parent_table_data.loc[next(parent_index_cycle)]
                for _ in range(next(freqs_cycle)):
                    this_fk_seed_df = pd.concat(
                        [this_fk_seed_df, pd.DataFrame([parent_record])]
                    ).reset_index(drop=True)

            seed_df = pd.concat(
                [
                    seed_df.reset_index(drop=True),
                    this_fk_seed_df.reset_index(drop=True),
                ],
                axis=1,
            )

        # We may have omitted some ancestral columns from training, so they must be omitted here as well.
        training_data = self.prepare_training_data(rel_data)[table]
        training_columns = list(training_data.columns)
        columns_to_drop = [
            col for col in seed_df.columns if col not in training_columns
        ]
        seed_df = seed_df.drop(columns=columns_to_drop)

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
        primary_key = rel_data.get_multigenerational_primary_key(table_name)
        if primary_key is None:
            return synthetic_table

        processed = synthetic_table
        processed[primary_key] = [i for i in range(len(synthetic_table))]

        foreign_key_maps = rel_data.get_ancestral_foreign_key_maps(table_name)
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
            table_name: rel_data.drop_ancestral_data(df)
            for table_name, df in output_tables.items()
        }

    def update_evaluation_from_model(
        self, evaluation: TableEvaluation, model: Model
    ) -> None:
        evaluation.cross_table_sqs = common.get_sqs_score(model)
        evaluation.cross_table_report_html = common.get_report_html(model)
        evaluation.cross_table_report_json = common.get_report_json(model)

    def update_evaluation_via_evaluate(
        self,
        evaluation: TableEvaluation,
        table: str,
        rel_data: RelationalData,
        synthetic_tables: Dict[str, pd.DataFrame],
    ) -> None:
        source_data = rel_data.get_table_data(table)
        synth_data = synthetic_tables[table]

        report = common.get_quality_report(
            source_data=source_data, synth_data=synth_data
        )

        evaluation.individual_sqs = report.peek().get("score")
        evaluation.individual_report_html = report.as_html
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
