import gzip
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pandas.testing as pdtest

import gretel_trainer.relational.ancestry as ancestry
from gretel_trainer.relational.core import TableEvaluation
from gretel_trainer.relational.strategies.single_table import SingleTableStrategy


def test_prepare_training_data_removes_primary_and_foreign_keys(pets):
    strategy = SingleTableStrategy()

    training_data = strategy.prepare_training_data(pets)

    assert set(training_data["pets"].columns) == {"name", "age"}


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
    job = strategy.get_generation_job("pets", pets, 2.0, {}, Path("/working"), [])

    assert job == {"params": {"num_records": 10}}


def test_uses_trained_model_to_update_individual_scores():
    strategy = SingleTableStrategy()
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

    assert evaluation.individual_sqs == 80
    assert evaluation.individual_report_json == {"report": "json"}

    assert evaluation.cross_table_sqs is None
    assert evaluation.cross_table_report_json is None


def test_falls_back_to_fetching_report_json_when_download_artifacts_fails():
    strategy = SingleTableStrategy()
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

    strategy = SingleTableStrategy()
    evaluation = TableEvaluation()

    mock_report = Mock()
    mock_report.peek = lambda: {"score": 85}
    mock_report.as_html = "HTML"
    mock_report.as_dict = {"REPORT": "JSON"}

    with patch(
        "gretel_trainer.relational.strategies.single_table.common.get_quality_report"
    ) as get_report:
        get_report.return_value = mock_report
        strategy.update_evaluation_via_evaluate(
            evaluation, "cities", rel_data, synthetic_tables
        )

    get_report.assert_called_once()

    ancestral_source_data = ancestry.get_table_data_with_ancestors(rel_data, "cities")
    ancestral_synth_data = ancestry.get_table_data_with_ancestors(
        synth_rel_data, "cities"
    )

    call_args = get_report.call_args

    pdtest.assert_frame_equal(ancestral_source_data, call_args.kwargs["source_data"])
    pdtest.assert_frame_equal(ancestral_synth_data, call_args.kwargs["synth_data"])

    assert evaluation.cross_table_sqs == 85
    assert evaluation.cross_table_report_json == {"REPORT": "JSON"}

    assert evaluation.individual_sqs is None
    assert evaluation.individual_report_json is None
