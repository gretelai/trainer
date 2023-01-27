import gzip
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pandas.testing as pdtest
import smart_open

import gretel_trainer.relational.ancestry as ancestry
from gretel_trainer.relational.core import TableEvaluation
from gretel_trainer.relational.strategies.independent import IndependentStrategy


def test_prepare_training_data_removes_primary_and_foreign_keys(pets):
    strategy = IndependentStrategy()

    training_data = strategy.prepare_training_data(pets)

    assert set(training_data["pets"].columns) == {"name", "age"}


def test_retraining_a_set_of_tables_only_retrains_those_tables(ecom):
    strategy = IndependentStrategy()
    assert set(strategy.tables_to_retrain(["users"], ecom)) == {"users"}
    assert set(strategy.tables_to_retrain(["users", "events"], ecom)) == {
        "users",
        "events",
    }
    assert set(strategy.tables_to_retrain(["products"], ecom)) == {"products"}


def test_table_generation_readiness(ecom):
    strategy = IndependentStrategy()

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
    strategy = IndependentStrategy()
    job = strategy.get_generation_job("pets", pets, 2.0, {}, Path("/working"), [])

    assert job == {"params": {"num_records": 10}}


def test_post_processing_one_to_one(pets):
    strategy = IndependentStrategy()

    # Models train on data with PKs and FKs removed,
    # so those fields won't be present in raw results
    raw_synth_tables = {
        "humans": pd.DataFrame(
            data={
                "name": ["Michael", "Dominique", "Dirk"],
                "city": ["Chicago", "Atlanta", "Dallas"],
            }
        ),
        "pets": pd.DataFrame(
            data={
                "name": ["Bull", "Hawk", "Maverick"],
                "age": [6, 0, 1],
            }
        ),
    }

    processed = strategy.post_process_synthetic_results(raw_synth_tables, [], pets)

    pdtest.assert_frame_equal(
        processed["humans"],
        pd.DataFrame(
            data={
                "name": ["Michael", "Dominique", "Dirk"],
                "city": ["Chicago", "Atlanta", "Dallas"],
                "id": [0, 1, 2],  # contiguous set of integers
            }
        ),
    )

    # FK order varies, so here we only assert on the deterministic fields
    pdtest.assert_frame_equal(
        processed["pets"][["name", "age", "id"]],
        pd.DataFrame(
            data={
                "name": ["Bull", "Hawk", "Maverick"],
                "age": [6, 0, 1],
                "id": [0, 1, 2],  # contiguous set of integers
            }
        ),
    )

    # Given 1:1 FK:PK relationship and record_size_ratio of 1,
    # we expect to see all PKs present in the FK column
    # (though we can't guarantee their order)
    assert set(processed["pets"]["human_id"]) == {0, 1, 2}


def test_post_processing_one_to_one_foreign_keys(pets):
    strategy = IndependentStrategy()

    raw_synth_tables = {
        "humans": pd.DataFrame(
            data={
                "name": ["Michael", "Dominique", "Dirk"],
                "city": ["Chicago", "Atlanta", "Dallas"],
            }
        ),
        "pets": pd.DataFrame(
            data={
                "name": ["Bull", "Hawk", "Maverick"],
                "age": [6, 0, 1],
            }
        ),
    }

    processed = strategy.post_process_synthetic_results(raw_synth_tables, [], pets)

    fk_values = set(processed["pets"]["human_id"])

    assert fk_values == {0, 1, 2}


def test_post_processing_foreign_keys_with_skewed_frequencies_and_different_size_tables(
    trips,
):
    strategy = IndependentStrategy()

    # Simulate a record_size_ratio of 1.5
    raw_synth_tables = {
        "vehicle_types": pd.DataFrame(
            data={"name": ["car", "train", "plane", "bus", "walk", "bike"]}
        ),
        "trips": pd.DataFrame(data={"purpose": ["w"] * 150}),
    }

    processed = strategy.post_process_synthetic_results(raw_synth_tables, [], trips)
    processed_trips = processed["trips"]

    fk_values = set(processed["trips"]["vehicle_type_id"])
    assert fk_values == {0, 1, 2, 3, 4, 5}

    fk_value_counts = defaultdict(int)
    for _, row in processed_trips.iterrows():
        fk_value = row["vehicle_type_id"]
        fk_value_counts[fk_value] = fk_value_counts[fk_value] + 1

    fk_value_counts = sorted(list(fk_value_counts.values()))

    assert fk_value_counts == [5, 5, 15, 30, 35, 60]


def test_uses_trained_model_to_update_individual_scores():
    strategy = IndependentStrategy()
    evaluations = {
        "table_1": TableEvaluation(),
        "table_2": TableEvaluation(),
    }
    model = Mock()

    with tempfile.TemporaryDirectory() as working_dir, patch(
        "gretel_trainer.relational.strategies.independent.common.download_artifacts"
    ) as download_artifacts, patch(
        "gretel_trainer.relational.strategies.independent.common.get_sqs_score"
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

    assert evaluation.individual_sqs == 80
    assert evaluation.individual_report_json == {"report": "json"}

    assert evaluation.cross_table_sqs is None
    assert evaluation.cross_table_report_json is None


def test_falls_back_to_fetching_report_json_when_download_artifacts_fails():
    strategy = IndependentStrategy()
    evaluations = {
        "table_1": TableEvaluation(),
        "table_2": TableEvaluation(),
    }
    model = Mock()

    with tempfile.TemporaryDirectory() as working_dir, patch(
        "gretel_trainer.relational.strategies.independent.common.download_artifacts"
    ) as download_artifacts, patch(
        "gretel_trainer.relational.strategies.independent.common.get_sqs_score"
    ) as get_sqs, patch(
        "gretel_trainer.relational.strategies.independent.common._get_report_json"
    ) as get_json:
        get_sqs.return_value = 80
        working_dir = Path(working_dir)
        download_artifacts.return_value = None
        get_json.return_value = {"report": "json"}

        strategy.update_evaluation_from_model(
            "table_1", evaluations, model, working_dir
        )

    evaluation = evaluations["table_1"]

    assert evaluation.individual_sqs == 80
    assert evaluation.individual_report_json == {"report": "json"}

    assert evaluation.cross_table_sqs is None
    assert evaluation.cross_table_report_json is None


def test_updates_cross_table_scores_using_evaluate(source_nba, synthetic_nba):
    rel_data, _, _, _ = source_nba
    synth_rel_data, synth_states, synth_cities, synth_teams = synthetic_nba
    synthetic_tables = {
        "teams": synth_teams,
        "cities": synth_cities,
        "states": synth_states,
    }

    strategy = IndependentStrategy()
    evaluation = TableEvaluation()

    mock_report = Mock()
    mock_report.peek = lambda: {"score": 85}
    mock_report.as_html = "HTML"
    mock_report.as_dict = {"REPORT": "JSON"}

    with tempfile.TemporaryDirectory() as working_dir, patch(
        "gretel_trainer.relational.strategies.independent.common.get_quality_report"
    ) as get_report:
        working_dir = Path(working_dir)
        get_report.return_value = mock_report
        strategy.update_evaluation_via_evaluate(
            evaluation, "cities", rel_data, synthetic_tables, working_dir
        )
        assert len(os.listdir(working_dir)) == 2
        assert (
            smart_open.open(working_dir / "expanded_evaluation_cities.html").read()
            == "HTML"
        )
        assert json.loads(
            smart_open.open(working_dir / "expanded_evaluation_cities.json").read()
        ) == {"REPORT": "JSON"}

    get_report.assert_called_once()

    ancestral_source_data = ancestry.get_table_data_with_ancestors(rel_data, "cities")
    ancestral_synth_data = ancestry.get_table_data_with_ancestors(
        synth_rel_data, "cities"
    )

    call_args = get_report.call_args
    call_args_kwargs = call_args[1]  # call_args.kwargs introduced in 3.8

    pdtest.assert_frame_equal(ancestral_source_data, call_args_kwargs["source_data"])
    pdtest.assert_frame_equal(ancestral_synth_data, call_args_kwargs["synth_data"])

    assert evaluation.cross_table_sqs == 85
    assert evaluation.cross_table_report_json == {"REPORT": "JSON"}

    assert evaluation.individual_sqs is None
    assert evaluation.individual_report_json is None
