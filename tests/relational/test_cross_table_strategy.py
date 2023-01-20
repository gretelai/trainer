import gzip
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pandas.testing as pdtest
import pytest

from gretel_trainer.relational.core import TableEvaluation
from gretel_trainer.relational.strategies.cross_table import CrossTableStrategy


def test_prepare_training_data_returns_multigenerational_data_without_keys_or_highly_unique_categorial_fields(
    pets,
):
    strategy = CrossTableStrategy()

    training_data = strategy.prepare_training_data(pets)

    assert set(training_data["pets"].columns) == {
        "self|id",
        "self|name",
        "self|age",
        "self|human_id",
        "self.human_id|id",
        "self.human_id|city",
        # self.human_id|name rejected (highly unique categorical)
    }


def test_retraining_a_set_of_tables_forces_retraining_descendants_as_well(ecom):
    strategy = CrossTableStrategy()
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
    strategy = CrossTableStrategy()

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
    strategy = CrossTableStrategy()

    # Table with no ancestors
    with tempfile.TemporaryDirectory() as tmp:
        working_dir = Path(tmp)
        parent_table_job = strategy.get_generation_job(
            "humans", pets, 2.0, {}, working_dir
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
            "pets", pets, 2.0, output_tables, working_dir
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


def test_post_processing_individual_synthetic_result(ecom):
    strategy = CrossTableStrategy()
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
    strategy = CrossTableStrategy()
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


def test_uses_trained_model_to_update_cross_table_scores():
    strategy = CrossTableStrategy()
    evaluations = {
        "table_1": TableEvaluation(),
        "table_2": TableEvaluation(),
    }
    model = Mock()

    with tempfile.TemporaryDirectory() as working_dir, patch(
        "gretel_trainer.relational.strategies.cross_table.common.download_artifacts"
    ) as download_artifacts, patch(
        "gretel_trainer.relational.strategies.cross_table.common.get_sqs_score"
    ) as get_sqs:
        get_sqs.return_value = 80
        working_dir = Path(working_dir)
        artifacts_subdir = working_dir / "artifacts_table_1"
        os.makedirs(artifacts_subdir, exist_ok=True)
        with gzip.open(str(artifacts_subdir / "report_json.json.gz"), "wb") as f:
            f.write(b'{"report": "json"}')
        download_artifacts.return_value = artifacts_subdir

        strategy.update_evaluation_from_model(
            "table_1", evaluations, model, working_dir
        )

    evaluation = evaluations["table_1"]

    assert evaluation.cross_table_sqs == 80
    assert evaluation.cross_table_report_json == {"report": "json"}

    assert evaluation.individual_sqs is None
    assert evaluation.individual_report_json is None


def test_falls_back_to_fetching_report_json_when_download_artifacts_fails():
    strategy = CrossTableStrategy()
    evaluations = {
        "table_1": TableEvaluation(),
        "table_2": TableEvaluation(),
    }
    model = Mock()

    with tempfile.TemporaryDirectory() as working_dir, patch(
        "gretel_trainer.relational.strategies.cross_table.common.download_artifacts"
    ) as download_artifacts, patch(
        "gretel_trainer.relational.strategies.cross_table.common.get_sqs_score"
    ) as get_sqs, patch(
        "gretel_trainer.relational.strategies.cross_table.common._get_report_json"
    ) as get_json:
        get_sqs.return_value = 80
        working_dir = Path(working_dir)
        download_artifacts.return_value = None
        get_json.return_value = {"report": "json"}

        strategy.update_evaluation_from_model(
            "table_1", evaluations, model, working_dir
        )

    evaluation = evaluations["table_1"]

    assert evaluation.cross_table_sqs == 80
    assert evaluation.cross_table_report_json == {"report": "json"}

    assert evaluation.individual_sqs is None
    assert evaluation.individual_report_json is None


def test_updates_single_table_scores_using_evaluate(source_nba, synthetic_nba):
    rel_data, _, source_cities, _ = source_nba
    _, synth_states, synth_cities, synth_teams = synthetic_nba
    synthetic_tables = {
        "teams": synth_teams,
        "cities": synth_cities,
        "states": synth_states,
    }

    strategy = CrossTableStrategy()
    evaluation = TableEvaluation()

    mock_report = Mock()
    mock_report.peek = lambda: {"score": 85}
    mock_report.as_html = "HTML"
    mock_report.as_dict = {"REPORT": "JSON"}

    with patch(
        "gretel_trainer.relational.strategies.cross_table.common.get_quality_report"
    ) as get_report:
        get_report.return_value = mock_report
        strategy.update_evaluation_via_evaluate(
            evaluation, "cities", rel_data, synthetic_tables
        )

    get_report.assert_called_once_with(
        source_data=source_cities, synth_data=synth_cities
    )

    assert evaluation.individual_sqs == 85
    assert evaluation.individual_report_json == {"REPORT": "JSON"}

    assert evaluation.cross_table_sqs is None
    assert evaluation.cross_table_report_json is None
