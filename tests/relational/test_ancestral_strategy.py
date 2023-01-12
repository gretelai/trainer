from unittest.mock import patch

import pandas as pd
import pandas.testing as pdtest
import pytest

from gretel_trainer.relational.strategies.ancestral import AncestralStrategy


def test_prepare_training_data_returns_multigenerational_data_without_keys_or_highly_unique_categorial_fields(
    pets,
):
    strategy = AncestralStrategy()

    training_pets = strategy.prepare_training_data("pets", pets)

    assert set(training_pets.columns) == {
        "self|id",
        "self|name",
        "self|age",
        "self|human_id",
        "self.human_id|id",
        "self.human_id|city",
        # self.human_id|name rejected (highly unique categorical)
    }


def test_retraining_a_set_of_tables_forces_retraining_descendants_as_well(ecom):
    strategy = AncestralStrategy()
    assert set(strategy.tables_to_retrain(["users"], ecom)) == {
        "users",
        "events",
        "order_items",
    }
    assert set(strategy.tables_to_retrain(["products"], ecom)) == {
        "products",
        "inventory_items",
        "order_items",
    }
    assert set(strategy.tables_to_retrain(["users", "products"], ecom)) == {
        "users",
        "events",
        "products",
        "inventory_items",
        "order_items",
    }


def test_table_generation_readiness(ecom):
    strategy = AncestralStrategy()

    # To start, "eldest generation" tables (those with no parents / outbound foreign keys) are ready
    assert set(strategy.ready_to_generate(ecom, [], [])) == {
        "users",
        "distribution_center",
    }

    # Once a table has been started, it is no longer ready
    assert set(strategy.ready_to_generate(ecom, ["users"], [])) == {
        "distribution_center"
    }

    # It's possible to be in a state where work is happening but nothing is ready
    assert (
        set(strategy.ready_to_generate(ecom, ["users", "distribution_center"], []))
        == set()
    )

    # `events` was only blocked by `users`; once the latter completes, the former is ready,
    # regardless of the state of the unrelated `distribution_center` table
    assert set(
        strategy.ready_to_generate(ecom, ["distribution_center"], ["users"])
    ) == {"events"}

    # Similarly, the completion of `distribution_center` unblocks `products`,
    # regardless of progress on `events`
    assert set(
        strategy.ready_to_generate(ecom, [], ["users", "distribution_center"])
    ) == {"events", "products"}

    # Remaining tables become ready as their parents complete
    assert set(
        strategy.ready_to_generate(
            ecom, [], ["users", "distribution_center", "events", "products"]
        )
    ) == {"inventory_items"}

    # As above, being in progress is not enough! Work is happening but nothing new is ready
    assert (
        set(
            strategy.ready_to_generate(
                ecom,
                ["inventory_items"],
                ["users", "distribution_center", "events", "products"],
            )
        )
        == set()
    )

    assert set(
        strategy.ready_to_generate(
            ecom,
            [],
            ["users", "distribution_center", "events", "products", "inventory_items"],
        )
    ) == {"order_items"}

    assert (
        set(
            strategy.ready_to_generate(
                ecom,
                ["order_items"],
                [
                    "users",
                    "distribution_center",
                    "events",
                    "products",
                    "inventory_items",
                ],
            )
        )
        == set()
    )
    assert (
        set(
            strategy.ready_to_generate(
                ecom,
                [],
                [
                    "users",
                    "distribution_center",
                    "events",
                    "products",
                    "inventory_items",
                    "order_items",
                ],
            )
        )
        == set()
    )


def test_generation_job(pets):
    strategy = AncestralStrategy()

    # Table with no ancestors
    parent_table_job = strategy.get_generation_job("humans", pets, 2.0, {})
    assert parent_table_job == {"params": {"num_records": 10}}

    # Table with ancestors
    synthetic_humans = pd.DataFrame(
        data={
            "self|name": [
                "Miles Davis",
                "Wayne Shorter",
                "Herbie Hancock",
                "Ron Carter",
                "Tony Williams",
            ],
            "self|city": [
                "New York",
                "New York",
                "New York",
                "New York",
                "Los Angeles",
            ],
            "self|id": [1, 2, 3, 4, 5],
        }
    )
    child_table_job = strategy.get_generation_job(
        "pets", pets, 2.0, {"humans": synthetic_humans}
    )

    expected_seed_df = pd.DataFrame(
        data={
            "self.human_id|name": [
                "Miles Davis",
                "Wayne Shorter",
                "Herbie Hancock",
                "Ron Carter",
                "Tony Williams",
            ],
            "self.human_id|city": [
                "New York",
                "New York",
                "New York",
                "New York",
                "Los Angeles",
            ],
            "self.human_id|id": [1, 2, 3, 4, 5],
        }
    )

    assert set(child_table_job.keys()) == {"data_source"}
    pdtest.assert_frame_equal(child_table_job["data_source"], expected_seed_df)


def test_generation_jobs(pets):
    strategy = AncestralStrategy()

    # Table with no ancestors
    parent_table_jobs = strategy.get_generation_jobs("humans", pets, 2.0, {})

    assert len(parent_table_jobs) == 1
    assert parent_table_jobs[0] == {"num_records": 10}

    # Table with ancestors
    synthetic_humans = pd.DataFrame(
        data={
            "self|name": [
                "Miles Davis",
                "Wayne Shorter",
                "Herbie Hancock",
                "Ron Carter",
                "Tony Williams",
            ],
            "self|city": [
                "New York",
                "New York",
                "New York",
                "New York",
                "Los Angeles",
            ],
            "self|id": [1, 2, 3, 4, 5],
        }
    )
    child_table_jobs = strategy.get_generation_jobs(
        "pets", pets, 2.0, {"humans": synthetic_humans}
    )

    expected_seed_df = pd.DataFrame(
        data={
            "self.human_id|name": [
                "Miles Davis",
                "Wayne Shorter",
                "Herbie Hancock",
                "Ron Carter",
                "Tony Williams",
            ],
            "self.human_id|city": [
                "New York",
                "New York",
                "New York",
                "New York",
                "Los Angeles",
            ],
            "self.human_id|id": [1, 2, 3, 4, 5],
        }
    )

    assert len(child_table_jobs) == 1
    pdtest.assert_frame_equal(child_table_jobs[0]["seed_df"], expected_seed_df)


def test_collecting_results_returns_lone_result(pets):
    strategy = AncestralStrategy()
    pets_data = pets.get_table_data("pets")

    result = strategy.collect_generation_results([pets_data], "pets", pets)

    pdtest.assert_frame_equal(result, pets_data)


def test_evalute(ecom):
    strategy = AncestralStrategy()
    synthetic_tables = {"users": pd.DataFrame()}

    with patch(
        "gretel_trainer.relational.strategies.ancestral.get_sqs_via_evaluate"
    ) as get_sqs:
        get_sqs.return_value = 80
        table_evaluation = strategy.evaluate("users", ecom, 90, synthetic_tables)

    # The model is trained on ancestral data, so its SQS score is the ancestral SQS
    assert table_evaluation.ancestral_sqs == 90
    # The individual score comes from the Evaluate API
    assert table_evaluation.individual_sqs == 80
