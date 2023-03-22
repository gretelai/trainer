import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pandas.testing as pdtest
import pytest
import smart_open

import gretel_trainer.relational.ancestry as ancestry
from gretel_trainer.relational.core import MultiTableException, TableEvaluation
from gretel_trainer.relational.strategies.ancestral import AncestralStrategy


def test_preparing_training_data_does_not_mutate_source_data(pets, art):
    for rel_data in [pets, art]:
        original_tables = {
            table: rel_data.get_table_data(table).copy()
            for table in rel_data.list_all_tables()
        }

        strategy = AncestralStrategy()
        strategy.prepare_training_data(rel_data)

        for table in rel_data.list_all_tables():
            pdtest.assert_frame_equal(
                original_tables[table], rel_data.get_table_data(table)
            )


def test_prepare_training_data_returns_multigenerational_data(pets):
    strategy = AncestralStrategy()

    training_data = strategy.prepare_training_data(pets)

    for expected_column in ["self|id", "self|name", "self.human_id|id"]:
        assert expected_column in training_data["pets"]


def test_prepare_training_data_drops_highly_unique_categorical_ancestor_fields(art):
    art.update_table_data(
        table="artists",
        data=pd.DataFrame(
            data={
                "id": [f"A{i}" for i in range(100)],
                "name": [str(i) for i in range(100)],
            }
        ),
    )
    art.update_table_data(
        table="paintings",
        data=pd.DataFrame(
            data={
                "id": [f"P{i}" for i in range(100)],
                "artist_id": [f"A{i}" for i in range(100)],
                "name": [str(i) for i in range(100)],
            }
        ),
    )

    strategy = AncestralStrategy()
    training_data = strategy.prepare_training_data(art)

    # Does not contain `self.artist_id|name` because it is highly unique categorical
    assert set(training_data["paintings"].columns) == {
        "self|id",
        "self|name",
        "self|artist_id",
        "self.artist_id|id",
    }


def test_prepare_training_data_drops_highly_nan_ancestor_fields(art):
    highly_nan_names = []
    for i in range(100):
        if i > 70:
            highly_nan_names.append(None)
        else:
            highly_nan_names.append("some name")
    art.update_table_data(
        table="artists",
        data=pd.DataFrame(
            data={
                "id": [f"A{i}" for i in range(100)],
                "name": highly_nan_names,
            }
        ),
    )
    art.update_table_data(
        table="paintings",
        data=pd.DataFrame(
            data={
                "id": [f"P{i}" for i in range(100)],
                "artist_id": [f"A{i}" for i in range(100)],
                "name": [str(i) for i in range(100)],
            }
        ),
    )

    strategy = AncestralStrategy()
    training_data = strategy.prepare_training_data(art)

    # Does not contain `self.artist_id|name` because it is highly NaN
    assert set(training_data["paintings"].columns) == {
        "self|id",
        "self|name",
        "self|artist_id",
        "self.artist_id|id",
    }


def test_prepare_training_data_translates_alphanumeric_keys_and_adds_min_max_records(
    art,
):
    strategy = AncestralStrategy()
    training_data = strategy.prepare_training_data(art)

    # Artists, a parent table, should have 1 additional row
    assert len(training_data["artists"]) == len(art.get_table_data("artists")) + 1
    # The last record has the artifical max PK
    assert training_data["artists"]["self|id"].to_list() == [0, 1, 2, 3, 200]
    # We do not assert on the value of "self|name" because the artificial max PK record is
    # randomly sampled from source and so the exact value is not deterministic

    # Paintings, as a child table, should have 3 additional rows
    # - artificial max PK
    # - artificial min FKs
    # - artificial max FKs
    assert len(training_data["paintings"]) == len(art.get_table_data("paintings")) + 3

    last_three = training_data["paintings"].tail(3)
    last_two = last_three.tail(2)

    # PKs are max, +1, +2
    assert last_three["self|id"].to_list() == [350, 351, 352]
    # FKs on last two rows (artifical FKs) are min, max
    assert last_two["self|artist_id"].to_list() == [0, 200]
    assert last_two["self.artist_id|id"].to_list() == [0, 200]


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


def test_preserve_tables(ecom):
    strategy = AncestralStrategy()

    with pytest.raises(MultiTableException):
        # Need to also preserve parent users
        strategy.validate_preserved_tables(["events"], ecom)

    with pytest.raises(MultiTableException):
        # Need to also preserve parent products
        strategy.validate_preserved_tables(
            ["distribution_center", "inventory_items"], ecom
        )

    assert strategy.validate_preserved_tables(["users", "events"], ecom) is None
    assert (
        strategy.validate_preserved_tables(
            ["distribution_center", "products", "inventory_items"], ecom
        )
        is None
    )


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

    training_columns = {
        "humans": [
            "self|id",
            "self|name",
            "self|city",
        ],
        "pets": [
            "self|id",
            "self|name",
            "self|age",
            "self|human_id",
            "self.human_id|id",
            # "self.human_id|name", # highly unique categorical
            "self.human_id|city",
        ],
    }

    # Table with no ancestors
    with tempfile.TemporaryDirectory() as tmp:
        working_dir = Path(tmp)
        parent_table_job = strategy.get_generation_job(
            "humans", pets, 2.0, {}, working_dir, training_columns["humans"]
        )
        assert len(os.listdir(working_dir)) == 0
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
    output_tables = {"humans": synthetic_humans}
    with tempfile.TemporaryDirectory() as tmp:
        working_dir = Path(tmp)
        child_table_job = strategy.get_generation_job(
            "pets", pets, 2.0, output_tables, working_dir, training_columns["pets"]
        )

        assert len(os.listdir(working_dir)) == 1
        assert set(child_table_job.keys()) == {"data_source"}
        child_table_seed_df = pd.read_csv(child_table_job["data_source"])

    # `self.human_id|name` should not be present in seed because it was
    # excluded from training data (highly-unique categorical field)
    expected_seed_df = pd.DataFrame(
        data={
            "self.human_id|city": [
                "New York",
                "New York",
                "New York",
                "New York",
                "Los Angeles",
                "New York",
                "New York",
                "New York",
                "New York",
                "Los Angeles",
            ],
            "self.human_id|id": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        }
    )

    pdtest.assert_frame_equal(child_table_seed_df, expected_seed_df)

    # sanity check: assert we did not mutate the originally-supplied synthetic tables
    assert set(output_tables["humans"].columns) == {"self|name", "self|city", "self|id"}


def test_generation_job_seeds_go_back_multiple_generations(source_nba, synthetic_nba):
    source_nba = source_nba[0]
    synthetic_nba = synthetic_nba[0]
    output_tables = {
        "cities": ancestry.get_table_data_with_ancestors(synthetic_nba, "cities"),
        "states": ancestry.get_table_data_with_ancestors(synthetic_nba, "states"),
    }
    training_columns = {
        "teams": [
            "self|name",
            "self|id",
            "self|city_id",
            "self.city_id|id",
            "self.city_id|state_id",
            # "self.city_id|name", # highly unique categorical
            "self.city_id.state_id|id",
            # "self.city_id.state_id|name", # highly unique categorical
        ],
        "cities": [
            "self|id",
            "self|state_id",
            # "self|name", # highly unique categorical
            "self.state_id|id",
            # "self.state_id|name", # highly unique categorical
        ],
        "states": [
            "self|id",
            "self|name",
        ],
    }

    strategy = AncestralStrategy()

    with tempfile.TemporaryDirectory() as tmp:
        working_dir = Path(tmp)
        job = strategy.get_generation_job(
            "teams",
            source_nba,
            1.0,
            output_tables,
            working_dir,
            training_columns["teams"],
        )
        seed_df = pd.read_csv(job["data_source"])

    expected_seed_df_columns = {
        "self.city_id|id",
        # "self.city_id|name", # highly unique categorical
        "self.city_id|state_id",
        "self.city_id.state_id|id",
        # "self.city_id.state_id|name", # highly unique categorical
    }

    assert set(seed_df.columns) == expected_seed_df_columns


def test_post_processing_individual_synthetic_result(ecom):
    strategy = AncestralStrategy()
    synth_events = pd.DataFrame(
        data={
            "self|id": [100, 101, 102, 103, 104],
            "self|user_id": [200, 201, 202, 203, 204],
            "self.user_id|id": [10, 11, 12, 13, 14],
        }
    )

    processed_events = strategy.post_process_individual_synthetic_result(
        "events", ecom, synth_events
    )

    expected_post_processing = pd.DataFrame(
        data={
            "self|id": [0, 1, 2, 3, 4],
            "self|user_id": [10, 11, 12, 13, 14],
            "self.user_id|id": [10, 11, 12, 13, 14],
        }
    )

    pdtest.assert_frame_equal(expected_post_processing, processed_events)


def test_post_process_synthetic_results(ecom):
    strategy = AncestralStrategy()
    out_events = pd.DataFrame(
        data={
            "self|id": [0, 1, 2],
            "self|browser": ["chrome", "safari", "brave"],
            "self|traffic_source": ["mobile", "mobile", "mobile"],
            "self|user_id": [0, 1, 2],
            "self.user_id|id": [0, 1, 2],
            "self.user_id|first_name": ["a", "b", "c"],
            "self.user_id|last_name": ["A", "B", "C"],
            "self.user_id|ssn": ["111", "222", "333"],
        }
    )
    out_users = pd.DataFrame(
        data={
            "self|id": [0, 1, 2],
            "self|first_name": ["a", "b", "c"],
            "self|last_name": ["A", "B", "C"],
            "self|ssn": ["111", "222", "333"],
        }
    )
    output_tables = {
        "events": out_events,
        "users": out_users,
    }

    processed_tables = strategy.post_process_synthetic_results(output_tables, [], ecom)

    expected_events = pd.DataFrame(
        data={
            "id": [0, 1, 2],
            "browser": ["chrome", "safari", "brave"],
            "traffic_source": ["mobile", "mobile", "mobile"],
            "user_id": [0, 1, 2],
        }
    )
    expected_users = pd.DataFrame(
        data={
            "id": [0, 1, 2],
            "first_name": ["a", "b", "c"],
            "last_name": ["A", "B", "C"],
            "ssn": ["111", "222", "333"],
        }
    )

    pdtest.assert_frame_equal(expected_events, processed_tables["events"])
    pdtest.assert_frame_equal(expected_users, processed_tables["users"])


def test_uses_trained_model_to_update_cross_table_scores(report_json_dict):
    strategy = AncestralStrategy()
    evaluations = {
        "table_1": TableEvaluation(),
        "table_2": TableEvaluation(),
    }
    model = Mock()

    with tempfile.TemporaryDirectory() as working_dir, patch(
        "gretel_trainer.relational.strategies.ancestral.common.download_artifacts"
    ) as download_artifacts:
        working_dir = Path(working_dir)
        with open(
            working_dir / "synthetics_cross_table_evaluation_table_1.json", "w"
        ) as f:
            f.write(json.dumps(report_json_dict))

        strategy.update_evaluation_from_model(
            "table_1", evaluations, model, working_dir
        )

    evaluation = evaluations["table_1"]

    assert evaluation.cross_table_sqs == 95
    assert evaluation.cross_table_report_json == report_json_dict

    assert evaluation.individual_sqs is None
    assert evaluation.individual_report_json is None


def test_falls_back_to_fetching_report_json_when_download_artifacts_fails(
    report_json_dict,
):
    strategy = AncestralStrategy()
    evaluations = {
        "table_1": TableEvaluation(),
        "table_2": TableEvaluation(),
    }
    model = Mock()

    with tempfile.TemporaryDirectory() as working_dir, patch(
        "gretel_trainer.relational.strategies.ancestral.common.download_artifacts"
    ) as download_artifacts, patch(
        "gretel_trainer.relational.strategies.ancestral.common._get_report_json"
    ) as get_json:
        working_dir = Path(working_dir)
        download_artifacts.return_value = None
        get_json.return_value = report_json_dict

        strategy.update_evaluation_from_model(
            "table_1", evaluations, model, working_dir
        )

    evaluation = evaluations["table_1"]

    assert evaluation.cross_table_sqs == 95
    assert evaluation.cross_table_report_json == report_json_dict

    assert evaluation.individual_sqs is None
    assert evaluation.individual_report_json is None
