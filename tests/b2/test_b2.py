import os
import tempfile
from unittest.mock import Mock, patch

import pandas as pd
import pandas.testing as pdtest
import pytest
from gretel_client.projects.jobs import Status

from gretel_trainer.b2 import Datatype, GretelGPTX, GretelLSTM, compare, make_dataset
from tests.b2.mocks import DoNothingModel, FailsToGenerate, FailsToTrain, TailoredActgan


def test_run_with_gretel_dataset(working_dir, project, evaluate_report_path, iris):
    evaluate_model = Mock(
        status=Status.COMPLETED,
    )
    evaluate_model.get_artifact_link.return_value = evaluate_report_path
    project.create_model_obj.side_effect = [evaluate_model]

    comparison = compare(
        datasets=[iris],
        models=[DoNothingModel],
        working_dir=working_dir,
    ).wait()

    assert len(comparison.results) == 1
    result = comparison.results.iloc[0]
    _iris_shape = pd.read_csv(iris.data_source).shape
    assert result["Input data"] == "iris"
    assert result["Model"] == "DoNothingModel"
    assert result["Rows"] == _iris_shape[0]
    assert result["Columns"] == _iris_shape[1]
    assert result["Status"] == "Completed"
    assert result["SQS"] == 95


def test_run_with_custom_csv_dataset(working_dir, project, evaluate_report_path, df):
    evaluate_model = Mock(
        status=Status.COMPLETED,
    )
    evaluate_model.get_artifact_link.return_value = evaluate_report_path
    project.create_model_obj.side_effect = [evaluate_model]

    with tempfile.NamedTemporaryFile() as f:
        df.to_csv(f.name, index=False)

        dataset = make_dataset(f.name, datatype="tabular", name="pets")

        comparison = compare(
            datasets=[dataset],
            models=[DoNothingModel],
            working_dir=working_dir,
        ).wait()

    assert len(comparison.results) == 1
    result = comparison.results.iloc[0]
    assert result["Input data"] == "pets"
    assert result["Model"] == "DoNothingModel"
    assert result["Rows"] == 3
    assert result["Columns"] == 2
    assert result["Status"] == "Completed"
    assert result["SQS"] == 95


def test_run_with_custom_dataframe_dataset(
    working_dir, project, evaluate_report_path, df
):
    evaluate_model = Mock(
        status=Status.COMPLETED,
    )
    evaluate_model.get_artifact_link.return_value = evaluate_report_path
    project.create_model_obj.side_effect = [evaluate_model]

    dataset = make_dataset(df, datatype="tabular", name="pets")

    comparison = compare(
        datasets=[dataset],
        models=[DoNothingModel],
        working_dir=working_dir,
    ).wait()

    assert len(comparison.results) == 1
    result = comparison.results.iloc[0]
    assert result["Input data"] == "pets"
    assert result["Model"] == "DoNothingModel"
    assert result["Rows"] == 3
    assert result["Columns"] == 2
    assert result["Status"] == "Completed"
    assert result["SQS"] == 95

    working_dir_contents = os.listdir(working_dir)

    # The source dataframe is written to CSV in the working dir...
    assert "pets.csv" in working_dir_contents
    # ...as is the synthetic output CSV
    assert "synth_DoNothingModel-pets.csv" in working_dir_contents


@pytest.mark.parametrize("benchmark_model", [GretelLSTM, TailoredActgan])
def test_run_happy_path_gretel_sdk(
    benchmark_model, working_dir, iris, project, evaluate_report_path
):
    record_handler = Mock(
        status=Status.COMPLETED,
        billing_details={"total_time_seconds": 15},
    )

    model = Mock(
        status=Status.COMPLETED,
        billing_details={"total_time_seconds": 30},
    )
    model.create_record_handler_obj.return_value = record_handler

    evaluate_model = Mock(
        status=Status.COMPLETED,
    )
    evaluate_model.get_artifact_link.return_value = evaluate_report_path

    project.create_model_obj.side_effect = [model, evaluate_model]

    # Patch the pd.read_csv call that fetches record handler data
    mock_synth_data = pd.DataFrame(data={"synthetic": [1, 2], "data": [3, 4]})

    with patch("gretel_trainer.b2.gretel.strategy_sdk.pd.read_csv") as read_csv:
        read_csv.return_value = mock_synth_data
        comparison = compare(
            datasets=[iris],
            models=[benchmark_model],
            working_dir=working_dir,
        ).wait()

    assert len(comparison.results) == 1
    result = comparison.results.iloc[0]
    model_name = benchmark_model.__name__
    assert result["Model"] == model_name
    assert result["Status"] == "Completed"
    assert result["SQS"] == 95
    assert result["Train time (sec)"] == 30
    assert result["Generate time (sec)"] == 15
    assert result["Total time (sec)"] == 45

    # The synthetic data is written to the working directory
    working_dir_contents = os.listdir(working_dir)
    assert len(working_dir_contents) == 1
    filename = f"synth_{model_name}-iris.csv"
    assert filename in working_dir_contents
    df = pd.read_csv(f"{working_dir}/{filename}")
    pdtest.assert_frame_equal(df, mock_synth_data)


def test_sdk_model_failure(working_dir, iris, project):
    model = Mock(
        status=Status.ERROR,
        billing_details={"total_time_seconds": 30},
    )

    project.create_model_obj.side_effect = [model]

    comparison = compare(
        datasets=[iris],
        models=[GretelLSTM],
        working_dir=working_dir,
    ).wait()

    assert len(comparison.results) == 1
    result = comparison.results.iloc[0]
    assert result["Model"] == "GretelLSTM"
    assert result["Status"] == "Failed (train)"
    assert result["SQS"] is None
    assert result["Train time (sec)"] == 30
    assert result["Generate time (sec)"] is None
    assert result["Total time (sec)"] == 30


def test_run_with_failures(working_dir, project, iris):
    comparison = compare(
        datasets=[iris],
        models=[FailsToTrain, FailsToGenerate],
        working_dir=working_dir,
    ).wait()

    assert len(comparison.results) == 2
    assert set(comparison.results["Status"]) == {"Failed (train)", "Failed (generate)"}


def test_gptx_skips_too_many_columns(working_dir, project):
    two_columns = pd.DataFrame(
        data={"english": ["hello", "world"], "spanish": ["hola", "mundo"]}
    )
    dataset = make_dataset(
        two_columns, datatype=Datatype.natural_language, name="skippy"
    )

    comparison = compare(
        datasets=[dataset],
        models=[GretelGPTX],
        working_dir=working_dir,
    ).wait()

    assert len(comparison.results) == 1
    assert_is_skipped(comparison.results.iloc[0])


def test_gptx_skips_non_natural_language_datatype(working_dir, project):
    tabular = pd.DataFrame(data={"foo": [1, 2, 3]})
    dataset = make_dataset(tabular, datatype=Datatype.tabular, name="skippy")

    comparison = compare(
        datasets=[dataset],
        models=[GretelGPTX],
        working_dir=working_dir,
    ).wait()

    assert len(comparison.results) == 1
    assert_is_skipped(comparison.results.iloc[0])


def test_lstm_skips_datasets_with_over_150_columns(working_dir, project):
    jumbo = pd.DataFrame(columns=list(range(151)))
    dataset = make_dataset(jumbo, datatype=Datatype.tabular, name="skippy")

    comparison = compare(
        datasets=[dataset],
        models=[GretelLSTM],
        working_dir=working_dir,
    ).wait()

    assert len(comparison.results) == 1
    assert_is_skipped(comparison.results.iloc[0])


def assert_is_skipped(result):
    assert result["Status"] == "Skipped"
    assert result["SQS"] is None
    assert result["Train time (sec)"] is None
    assert result["Generate time (sec)"] is None
    assert result["Total time (sec)"] is None


def test_bad_session_exits_early(iris):
    class SomeException(Exception):
        pass

    with patch("gretel_trainer.b2.comparison.configure_session") as configure_session:
        configure_session.side_effect = SomeException()

        with pytest.raises(SomeException):
            compare(
                datasets=[iris],
                models=[GretelLSTM],
                working_dir="should_not_be_created",
            )

    assert not os.path.exists("should_not_be_created")
