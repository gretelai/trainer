from unittest.mock import Mock, patch

import pandas as pd
import pandas.testing as pdtest

from gretel_trainer.relational.core import TblEval
from gretel_trainer.relational.strategies.single_table import SingleTableStrategy


def test_prepare_training_data_removes_primary_and_foreign_keys(pets):
    strategy = SingleTableStrategy()

    training_pets = strategy.prepare_training_data("pets", pets)

    assert set(training_pets.columns) == {"name", "age"}


def test_retraining_a_set_of_tables_only_retrains_those_tables(ecom):
    strategy = SingleTableStrategy()
    assert set(strategy.tables_to_retrain(["users"], ecom)) == {"users"}
    assert set(strategy.tables_to_retrain(["users", "events"], ecom)) == {
        "users",
        "events",
    }
    assert set(strategy.tables_to_retrain(["products"], ecom)) == {"products"}


def test_table_generation_readiness(ecom):
    strategy = SingleTableStrategy()

    # All tables are immediately ready for generation
    assert set(strategy.ready_to_generate(ecom, [], [])) == {
        "users",
        "events",
        "distribution_center",
        "products",
        "inventory_items",
        "order_items",
    }

    # Tables that are in progress or finished are no longer ready
    assert set(strategy.ready_to_generate(ecom, ["users"], ["events"])) == {
        "distribution_center",
        "products",
        "inventory_items",
        "order_items",
    }


def test_generation_job_requests_num_records(pets):
    strategy = SingleTableStrategy()
    job = strategy.get_generation_job("pets", pets, 2.0, {})

    assert job == {"params": {"num_records": 10}}


def test_evalute(ecom):
    strategy = SingleTableStrategy()
    synthetic_tables = {"users": pd.DataFrame()}

    with patch(
        "gretel_trainer.relational.strategies.single_table.common.get_sqs_via_evaluate"
    ) as get_sqs:
        get_sqs.return_value = 80
        table_evaluation = strategy.evaluate("users", ecom, 90, synthetic_tables)

    # A model score was provided, so we only need to call Evaluate API once
    assert get_sqs.call_count == 1

    # The model is trained on individual table data, so its SQS score is the individual SQS
    assert table_evaluation.individual_sqs == 90
    # The cross-table score comes from the Evaluate API
    assert table_evaluation.cross_table_sqs == 80


def test_evaluate_without_model_score_calls_evaluate_twice(ecom):
    strategy = SingleTableStrategy()
    synthetic_tables = {"users": pd.DataFrame()}

    with patch(
        "gretel_trainer.relational.strategies.single_table.common.get_sqs_via_evaluate"
    ) as get_sqs:
        get_sqs.return_value = 80
        table_evaluation = strategy.evaluate("users", ecom, None, synthetic_tables)

    # A model score was not provided, so we need to call Evaluate API twice
    assert get_sqs.call_count == 2

    assert table_evaluation.individual_sqs == 80
    assert table_evaluation.cross_table_sqs == 80


def test_uses_trained_model_to_update_individual_scores():
    strategy = SingleTableStrategy()
    evaluation = TblEval()
    model = Mock()

    with patch(
        "gretel_trainer.relational.strategies.single_table.common.get_sqs_score"
    ) as get_sqs, patch(
        "gretel_trainer.relational.strategies.single_table.common.get_report_html"
    ) as get_html, patch(
        "gretel_trainer.relational.strategies.single_table.common.get_report_json"
    ) as get_json:
        get_sqs.return_value = 80
        get_html.return_value = "html"
        get_json.return_value = {"report": "json"}

        strategy.update_evaluation_from_model(evaluation, model)

    assert evaluation.individual_sqs == 80
    assert evaluation.individual_report_html == "html"
    assert evaluation.individual_report_json == {"report": "json"}

    assert evaluation.cross_table_sqs is None
    assert evaluation.cross_table_report_html is None
    assert evaluation.cross_table_report_json is None
