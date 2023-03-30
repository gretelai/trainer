import gzip
import os
import tempfile
from unittest.mock import Mock, patch

import pandas as pd
import pandas.testing as pdtest
import pytest
from gretel_client.projects.jobs import Status

from gretel_trainer.benchmark import (
    Comparison,
    Datatype,
    GretelGPTX,
    GretelLSTM,
    compare,
    make_dataset,
)
from gretel_trainer.benchmark.core import BenchmarkException
from tests.benchmark.mocks import (
    DoNothingModel,
    FailsToGenerate,
    FailsToTrain,
    SharedDictLstm,
    TailoredActgan,
)


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
    assert result["Status"] == "Complete"
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
    assert result["Status"] == "Complete"
    assert result["SQS"] == 95


def test_run_with_custom_psv_dataset(working_dir, project, evaluate_report_path, df):
    evaluate_model = Mock(
        status=Status.COMPLETED,
    )
    evaluate_model.get_artifact_link.return_value = evaluate_report_path
    project.create_model_obj.side_effect = [evaluate_model]

    with tempfile.NamedTemporaryFile() as f:
        df.to_csv(f.name, sep="|", index=False)

        dataset = make_dataset(f.name, datatype="tabular", name="pets", delimiter="|")

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
    assert result["Status"] == "Complete"
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
    assert result["Status"] == "Complete"
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

    with patch(
        "gretel_trainer.benchmark.gretel.strategy_sdk.GretelSDKStrategy._get_record_handler_data"
    ) as get:
        get.return_value = Mock(content=gzip.compress(b"synthetic,data\n1,3\n2,4"))
        comparison = compare(
            datasets=[iris],
            models=[benchmark_model],
            working_dir=working_dir,
        ).wait()

    assert len(comparison.results) == 1
    result = comparison.results.iloc[0]
    model_name = benchmark_model.__name__
    assert result["Model"] == model_name
    print(comparison.executors[f"{model_name}-iris"].exception)
    assert result["Status"] == "Complete"
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
    pdtest.assert_frame_equal(
        df, pd.DataFrame(data={"synthetic": [1, 2], "data": [3, 4]})
    )


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


def test_custom_gretel_model_configs_do_not_overwrite_each_other(
    working_dir, project, iris, df
):
    model = Mock(
        status=Status.ERROR,
        billing_details={"total_time_seconds": 30},
    )
    project.create_model_obj.return_value = model

    pets = make_dataset(df, datatype="tabular", name="pets")

    comparison = compare(
        datasets=[iris, pets],
        models=[SharedDictLstm],
        working_dir=working_dir,
    ).wait()

    model_names = [
        call.kwargs["model_config"]["name"]
        for call in project.create_model_obj.call_args_list
    ]

    assert set(model_names) == {"SharedDictLstm-iris", "SharedDictLstm-pets"}


def test_lengthy_names_with_trainer_throws_an_exception(working_dir, project, df):
    # Default project display name = 24 chars (benchmark-yyyymmddhhmmss)
    # GretelLSTM = 10
    # Remaining for dataset name = 45-24-10 = 11

    too_long = make_dataset(df, datatype="tabular", name="x" * 12)
    just_fits = make_dataset(df, datatype="tabular", name="x" * 11)

    with pytest.raises(BenchmarkException):
        Comparison(
            datasets=[too_long],
            models=[GretelLSTM],
            working_dir=working_dir,
            trainer=True,
        )

    Comparison(
        datasets=[just_fits],
        models=[GretelLSTM],
        working_dir=working_dir,
        trainer=True,
    )


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
