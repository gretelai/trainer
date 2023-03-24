import pandas as pd

from gretel_trainer.b2 import Datatype, GretelGPTX, GretelLSTM, compare, make_dataset
from tests.b2.mocks import DoNothingModel, FailsToGenerate, FailsToTrain


def test_run_with_gretel_dataset(iris):
    comparison = compare(
        datasets=[iris],
        models=[DoNothingModel],
    ).wait()

    assert len(comparison.results) == 1
    result = comparison.results.iloc[0]
    _iris_shape = pd.read_csv(iris.data_source).shape
    assert result["Input data"] == "iris"
    assert result["Model"] == "DoNothingModel"
    assert result["Rows"] == _iris_shape[0]
    assert result["Columns"] == _iris_shape[1]
    assert result["Status"] == "Completed"


def test_run_with_custom_csv_dataset(csv):
    dataset = make_dataset(csv, datatype="tabular", name="pets")

    comparison = compare(
        datasets=[dataset],
        models=[DoNothingModel],
    ).wait()

    assert len(comparison.results) == 1
    result = comparison.results.iloc[0]
    assert result["Input data"] == "pets"
    assert result["Model"] == "DoNothingModel"
    assert result["Rows"] == 3
    assert result["Columns"] == 2
    assert result["Status"] == "Completed"


def test_run_with_custom_dataframe_dataset(df):
    dataset = make_dataset(df, datatype="tabular", name="pets")

    comparison = compare(
        datasets=[dataset],
        models=[DoNothingModel],
    ).wait()

    assert len(comparison.results) == 1
    result = comparison.results.iloc[0]
    assert result["Input data"] == "pets"
    assert result["Model"] == "DoNothingModel"
    assert result["Rows"] == 3
    assert result["Columns"] == 2
    assert result["Status"] == "Completed"


# parametrize
def test_run_with_default_gretel_model_through_sdk():
    pass


def test_run_with_specific_gretel_model_config():
    pass


def test_run_with_failures(iris):
    comparison = compare(
        datasets=[iris],
        models=[FailsToTrain, FailsToGenerate],
    ).wait()

    assert len(comparison.results) == 2
    assert set(comparison.results["Status"]) == {"Failed (train)", "Failed (generate)"}


def test_gptx_skips_too_many_columns():
    two_columns = pd.DataFrame(
        data={"english": ["hello", "world"], "spanish": ["hola", "mundo"]}
    )
    dataset = make_dataset(
        two_columns, datatype=Datatype.natural_language, name="skippy"
    )

    comparison = compare(
        datasets=[dataset],
        models=[GretelGPTX],
    ).wait()

    assert len(comparison.results) == 1
    assert_is_skipped(comparison.results.iloc[0])


def test_gptx_skips_non_natural_language_datatype():
    tabular = pd.DataFrame(data={"foo": [1, 2, 3]})
    dataset = make_dataset(tabular, datatype=Datatype.tabular, name="skippy")

    comparison = compare(
        datasets=[dataset],
        models=[GretelGPTX],
    ).wait()

    assert len(comparison.results) == 1
    assert_is_skipped(comparison.results.iloc[0])


def test_lstm_skips_datasets_with_over_150_columns():
    jumbo = pd.DataFrame(columns=list(range(151)))
    dataset = make_dataset(jumbo, datatype=Datatype.tabular, name="skippy")

    comparison = compare(
        datasets=[dataset],
        models=[GretelLSTM],
    ).wait()

    assert len(comparison.results) == 1
    assert_is_skipped(comparison.results.iloc[0])


def assert_is_skipped(result):
    assert result["Status"] == "Skipped"
    assert result["SQS"] is None
    assert result["Train time (sec)"] is None
    assert result["Generate time (sec)"] is None
    assert result["Total time (sec)"] is None
