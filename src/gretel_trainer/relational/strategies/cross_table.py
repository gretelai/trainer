from collections import defaultdict
from typing import Any, Dict, List, Optional

import pandas as pd
from gretel_client.projects.models import Model
from pandas.api.types import is_string_dtype

import gretel_trainer.relational.strategies.common as common
from gretel_trainer.relational.core import (
    MultiTableException,
    RelationalData,
    TableEvaluation,
    TblEval,
)


class CrossTableStrategy:
    def __init__(self, model_type: str = "amplify"):
        if model_type not in ("amplify", "lstm"):
            raise MultiTableException(
                f"Unsupported model type: {model_type}. Supported model types are `Amplify` and `LSTM`."
            )
        self._model_type = model_type

    def prepare_training_data(
        self, table_name: str, rel_data: RelationalData
    ) -> pd.DataFrame:
        """
        Returns table data with all ancestor fields added,
        minus any highly-unique categorical fields from ancestors.
        """
        data = rel_data.get_table_data_with_ancestors(table_name)
        columns_to_drop = []

        for column in data.columns:
            if rel_data.is_ancestral_column(column) and _is_highly_unique_categorical(
                column, data
            ):
                columns_to_drop.append(column)

        return data.drop(columns=columns_to_drop)

    def get_seed_fields(
        self, table_name: str, rel_data: RelationalData
    ) -> Optional[List[str]]:
        seed_data = rel_data.build_seed_data_for_table(table_name)
        if seed_data is None:
            return None
        else:
            return [
                col for col in seed_data.columns if rel_data.is_ancestral_column(col)
            ]

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
        if len(rel_data.get_parents(table)) == 0:
            source_data_size = len(rel_data.get_table_data(table))
            synth_size = int(source_data_size * record_size_ratio)
            return {"params": {"num_records": synth_size}}
        else:
            seed_df = rel_data.build_seed_data_for_table(table, output_tables)
            return {"data_source": seed_df}

    def evaluate(
        self,
        table: str,
        rel_data: RelationalData,
        model_score: Optional[int],
        synthetic_tables: Dict[str, pd.DataFrame],
    ) -> TableEvaluation:
        if model_score is not None:
            cross_table_sqs = model_score
        else:
            cross_table_synthetic_data = rel_data.get_table_data_with_ancestors(
                table, synthetic_tables
            )
            cross_table_reference_data = rel_data.get_table_data_with_ancestors(table)
            cross_table_sqs = common.get_sqs_via_evaluate(
                cross_table_synthetic_data, cross_table_reference_data
            )

        individual_synthetic_data = synthetic_tables[table]
        individual_reference_data = rel_data.get_table_data(table)
        individual_sqs = common.get_sqs_via_evaluate(
            individual_synthetic_data, individual_reference_data
        )

        return TableEvaluation(
            individual_sqs=individual_sqs, cross_table_sqs=cross_table_sqs
        )

    def update_evaluation_from_model(self, evaluation: TblEval, model: Model) -> None:
        evaluation.cross_table_sqs = common.get_sqs_score(model)
        evaluation.cross_table_report_html = common.get_report_html(model)
        evaluation.cross_table_report_json = common.get_report_json(model)

    def update_evaluation_via_evaluate(
        self,
        evaluation: TblEval,
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
