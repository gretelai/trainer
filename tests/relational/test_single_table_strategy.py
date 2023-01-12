from unittest.mock import patch

import pandas as pd
import pandas.testing as pdtest

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


def test_yields_one_generation_job_using_num_records_param(pets):
    strategy = SingleTableStrategy()

    jobs = strategy.get_generation_jobs("pets", pets, 2.0, {})

    assert jobs == [{"num_records": 10}]


def test_collect_generation_results_returns_the_lone_output_dataframe(pets):
    strategy = SingleTableStrategy()
    pets_data = pets.get_table_data("pets")

    result = strategy.collect_generation_results([pets_data], "pets", pets)

    pdtest.assert_frame_equal(result, pets_data)


def test_evalute(ecom):
    strategy = SingleTableStrategy()
    synthetic_tables = {"users": pd.DataFrame()}

    with patch(
        "gretel_trainer.relational.strategies.single_table.get_sqs_via_evaluate"
    ) as get_sqs:
        get_sqs.return_value = 80
        table_evaluation = strategy.evaluate("users", ecom, 90, synthetic_tables)

    # The model is trained on individaul table data, so its SQS score is the individual SQS
    assert table_evaluation.individual_sqs == 90
    # The ancestral score comes from the Evaluate API
    assert table_evaluation.ancestral_sqs == 80
