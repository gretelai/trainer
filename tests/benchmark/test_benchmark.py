import gzip
import os
import tempfile

from unittest.mock import Mock, patch

import pandas as pd
import pandas.testing as pdtest
import pytest

from gretel_client.projects.jobs import Status
from gretel_client.projects.models import read_model_config
from gretel_trainer.benchmark import BenchmarkConfig, compare, create_dataset, Datatype
from gretel_trainer.benchmark.core import Dataset
from gretel_trainer.benchmark.gretel.models import GretelModel


class DoNothingModel:
    def train(self, source: Dataset, **kwargs) -> None:
        pass

    def generate(self, **kwargs) -> pd.DataFrame:
        return pd.DataFrame()


class FailsToTrain:
    def train(self, source: Dataset, **kwargs) -> None:
        raise Exception("failed")

    def generate(self, **kwargs) -> pd.DataFrame:
        return pd.DataFrame()


class FailsToGenerate:
    def train(self, source: Dataset, **kwargs) -> None:
        pass

    def generate(self, **kwargs) -> pd.DataFrame:
        raise Exception("failed")


class TailoredActgan(GretelModel):
    @property
    def config(self):
        c = read_model_config("synthetics/tabular-actgan")
        c["models"][0]["actgan"]["params"]["epochs"] = 100
        return c


def test_run_with_gretel_dataset(working_dir, project, evaluate_report_handle, iris):
    evaluate_model = Mock(
        status=Status.COMPLETED,
    )
    evaluate_model.get_artifact_handle.return_value = evaluate_report_handle
    project.create_model_obj.side_effect = [evaluate_model]

    session = compare(
        datasets=[iris],
        models=[DoNothingModel],
        config=BenchmarkConfig(
            working_dir=working_dir,
        ),
    ).wait()

    assert len(session.results) == 1
    result = session.results.iloc[0]
    _iris_shape = pd.read_csv(iris.data_source).shape
    assert result["Input data"] == "iris"
    assert result["Model"] == "DoNothingModel"
    assert result["Rows"] == _iris_shape[0]
    assert result["Columns"] == _iris_shape[1]
    assert result["Status"] == "Complete"
    assert result["SQS"] == 95


def test_run_with_custom_csv_dataset(working_dir, project, evaluate_report_handle, df):
    evaluate_model = Mock(
        status=Status.COMPLETED,
    )
    evaluate_model.get_artifact_handle.return_value = evaluate_report_handle
    project.create_model_obj.side_effect = [evaluate_model]

    with tempfile.NamedTemporaryFile() as f:
        df.to_csv(f.name, index=False)

        dataset = create_dataset(f.name, datatype="tabular", name="pets")

        session = compare(
            datasets=[dataset],
            models=[DoNothingModel],
            config=BenchmarkConfig(
                working_dir=working_dir,
            ),
        ).wait()

    assert len(session.results) == 1
    result = session.results.iloc[0]
    assert result["Input data"] == "pets"
    assert result["Model"] == "DoNothingModel"
    assert result["Rows"] == 3
    assert result["Columns"] == 2
    assert result["Status"] == "Complete"
    assert result["SQS"] == 95


def test_run_with_custom_psv_dataset(working_dir, project, evaluate_report_handle, df):
    evaluate_model = Mock(
        status=Status.COMPLETED,
    )
    evaluate_model.get_artifact_handle.return_value = evaluate_report_handle
    project.create_model_obj.side_effect = [evaluate_model]

    with tempfile.NamedTemporaryFile() as f:
        df.to_csv(f.name, sep="|", index=False)

        dataset = create_dataset(f.name, datatype="tabular", name="pets", delimiter="|")

        session = compare(
            datasets=[dataset],
            models=[DoNothingModel],
            config=BenchmarkConfig(
                working_dir=working_dir,
            ),
        ).wait()

    assert len(session.results) == 1
    result = session.results.iloc[0]
    assert result["Input data"] == "pets"
    assert result["Model"] == "DoNothingModel"
    assert result["Rows"] == 3
    assert result["Columns"] == 2
    assert result["Status"] == "Complete"
    assert result["SQS"] == 95


def test_run_with_custom_dataframe_dataset(
    working_dir, project, evaluate_report_handle, df
):
    evaluate_model = Mock(
        status=Status.COMPLETED,
    )
    evaluate_model.get_artifact_handle.return_value = evaluate_report_handle
    project.create_model_obj.side_effect = [evaluate_model]

    dataset = create_dataset(df, datatype="tabular", name="pets")

    session = compare(
        datasets=[dataset],
        models=[DoNothingModel],
        config=BenchmarkConfig(
            working_dir=working_dir,
        ),
    ).wait()

    assert len(session.results) == 1
    result = session.results.iloc[0]
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


@pytest.mark.parametrize("benchmark_model", [TailoredActgan])
def test_run_happy_path_gretel_sdk(
    benchmark_model, working_dir, iris, project, evaluate_report_handle
):
    record_handler = Mock(
        status=Status.COMPLETED,
        billing_details={"total_time_seconds": 15},
    )
    record_handler.submit_cloud.return_value = record_handler

    model = Mock(
        status=Status.COMPLETED,
        billing_details={"total_time_seconds": 30},
    )
    model.create_record_handler_obj.return_value = record_handler
    model.submit.return_value = model

    evaluate_model = Mock(
        status=Status.COMPLETED,
    )
    evaluate_model.get_artifact_handle.return_value = evaluate_report_handle

    project.create_model_obj.side_effect = [model, evaluate_model]

    with patch(
        "gretel_trainer.benchmark.gretel.strategy_sdk.GretelSDKStrategy._get_record_handler_data"
    ) as get:
        get.return_value = Mock(content=gzip.compress(b"synthetic,data\n1,3\n2,4"))
        session = compare(
            datasets=[iris],
            models=[benchmark_model],
            config=BenchmarkConfig(
                working_dir=working_dir,
            ),
        ).wait()

    assert len(session.results) == 1
    result = session.results.iloc[0]
    model_name = benchmark_model.__name__
    assert result["Model"] == model_name
    assert result["Status"] == "Complete"
    assert result["SQS"] == 95
    assert result["Train time (sec)"] == 30
    assert result["Generate time (sec)"] == 15
    assert result["Total time (sec)"] == 45

    working_dir_contents = os.listdir(working_dir)
    assert len(working_dir_contents) == 2

    # The synthetic data is written to the working directory
    filename = f"synth_{model_name}-iris.csv"
    assert filename in working_dir_contents
    df = pd.read_csv(f"{working_dir}/{filename}")
    pdtest.assert_frame_equal(
        df, pd.DataFrame(data={"synthetic": [1, 2], "data": [3, 4]})
    )

    # Snapshot results are written to the working directory
    filename = f"{model_name}-iris_result_data.csv"
    assert filename in working_dir_contents
    df = pd.read_csv(f"{working_dir}/{filename}")
    pdtest.assert_frame_equal(df, session.results)


def test_run_with_failures(working_dir, project, iris):
    session = compare(
        datasets=[iris],
        models=[FailsToTrain, FailsToGenerate],
        config=BenchmarkConfig(
            working_dir=working_dir,
        ),
    ).wait()

    assert len(session.results) == 2
    assert set(session.results["Status"]) == {"Failed (train)", "Failed (generate)"}


def assert_is_skipped(result):
    assert result["Status"] == "Skipped"
    assert result["SQS"] is None
    assert result["Train time (sec)"] is None
    assert result["Generate time (sec)"] is None
    assert result["Total time (sec)"] is None
