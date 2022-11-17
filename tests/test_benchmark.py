import os
import shutil
import uuid

from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from typing import Union
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from gretel_trainer import models as tm

from gretel_trainer.benchmark import (
    Datatype,
    GretelAmplify,
    GretelAuto,
    GretelACTGAN,
    GretelGPTX,
    GretelLSTM,
    GretelModel,
)
from gretel_trainer.benchmark.compare import compare, RuntimeConfig
from gretel_trainer.benchmark.core import BenchmarkException
from gretel_trainer.benchmark.custom.datasets import make_dataset
from gretel_trainer.benchmark.gretel.datasets import GretelPublicDatasetRepo
from gretel_trainer.benchmark.gretel.sdk import GretelSDK

from tests.mocks import (
    DictConfigGretelModel,
    DoNothingModel,
    FailingModel,
    LocalFileConfigGretelModel,
    mock_gretel_trainer_factory,
    MockGretelTrainer,
)

TEST_BENCHMARK_DIR = "./.benchmark_test"

TEST_GRETEL_DATASET_REPO = GretelPublicDatasetRepo(
    bucket="gretel-datasets",
    region="us-west-2",
    load_dir=TEST_BENCHMARK_DIR,
)

TEST_RUNTIME_CONFIG = RuntimeConfig(
    local_dir=TEST_BENCHMARK_DIR,
    project_prefix="benchmark-proj",
    thread_pool=ThreadPoolExecutor(4),
    wait_secs=0.1,
    auto_clean=True,
)


def _make_dataset(
    sources,
    datatype: Union[Datatype, str] = Datatype.TABULAR_MIXED,
    delimiter=",",
    namespace=None,
):
    return make_dataset(
        sources,
        datatype=datatype,
        delimiter=delimiter,
        namespace=namespace,
        local_dir=TEST_BENCHMARK_DIR,
    )


def _make_gretel_sdk(
    configure_session=None,
    create_project=None,
    search_projects=None,
    evaluate=None,
    poll=None,
) -> GretelSDK:
    return GretelSDK(
        configure_session=configure_session or Mock(),
        create_project=create_project or Mock(),
        search_projects=search_projects or Mock(return_value=[Mock()]),
        evaluate=evaluate or Mock(return_value=42),
        poll=poll or Mock(),
    )


def test_end_to_end_with_custom_datasets(df, csv, psv):
    df_dataset = _make_dataset([df, df.transpose()], datatype=Datatype.TABULAR_MIXED)
    csv_dataset = _make_dataset([csv], datatype=Datatype.TABULAR_MIXED)
    psv_dataset = _make_dataset(
        [psv], namespace="pipes", delimiter="|", datatype=Datatype.TABULAR_MIXED
    )

    comparison = compare(
        datasets=[
            df_dataset,
            csv_dataset,
            psv_dataset,
        ],
        models=[
            DoNothingModel,
            GretelAuto,
        ],
        runtime_config=TEST_RUNTIME_CONFIG,
        gretel_sdk=_make_gretel_sdk(),
        gretel_trainer_factory=mock_gretel_trainer_factory(get_sqs_score=84),
    ).wait()

    def _unique_results(col: str):
        unique_results = list(set(comparison.results[col].values.tolist()))
        unique_results.sort()
        return unique_results

    assert len(comparison.results) == 8
    assert _unique_results("Input data") == [
        f"{csv}",
        "DataFrames::0",
        "DataFrames::1",
        f"pipes::{psv}",
    ]
    assert _unique_results("Model") == ["DoNothingModel", "GretelAuto"]
    assert _unique_results("Status") == ["Completed"]
    assert _unique_results("Rows") == [2, 3]
    assert _unique_results("Columns") == [2, 3]
    assert _unique_results("SQS") == [42, 84]


def test_failures_during_train_generate_or_custom_evaluate(csv):
    def _fail(synthetic, reference):
        raise Exception("failed")

    csv_dataset = _make_dataset([csv])

    comparison = compare(
        datasets=[csv_dataset],
        models=[
            FailingModel.during_train,
            FailingModel.during_generate,
            DoNothingModel,
        ],
        runtime_config=TEST_RUNTIME_CONFIG,
        gretel_sdk=_make_gretel_sdk(evaluate=_fail),
        gretel_trainer_factory=mock_gretel_trainer_factory(),
    ).wait()

    assert comparison.results["Status"].values.tolist() == [
        "Failed (train)",
        "Failed (generate)",
        "Failed (evaluate)",
    ]


def test_failures_during_cleanup_are_ignored(csv):
    def _failing_search(_):
        raise Exception("failed")

    csv_dataset = _make_dataset([csv])

    comparison = compare(
        datasets=[csv_dataset],
        models=[
            GretelAuto,
        ],
        runtime_config=TEST_RUNTIME_CONFIG,
        gretel_sdk=_make_gretel_sdk(search_projects=_failing_search),
        gretel_trainer_factory=mock_gretel_trainer_factory(),
    ).wait()

    assert comparison.results["Status"].values.tolist() == ["Completed"]


@pytest.mark.parametrize(
    "model,expected_model_type",
    [
        (GretelAuto, None),
        (GretelACTGAN, tm.GretelACTGAN),
        (GretelLSTM, tm.GretelLSTM),
        (GretelAmplify, tm.GretelAmplify),
        (DictConfigGretelModel, tm.GretelLSTM),
        (LocalFileConfigGretelModel, tm.GretelLSTM),
    ],
)
def test_models_using_trainer_executor(model, expected_model_type, csv):
    mock_gretel_trainer = MockGretelTrainer()

    csv_dataset = _make_dataset([csv])

    compare(
        datasets=[csv_dataset],
        models=[model],
        runtime_config=TEST_RUNTIME_CONFIG,
        gretel_sdk=_make_gretel_sdk(),
        gretel_trainer_factory=lambda **kw: mock_gretel_trainer._factory_args(**kw),
    ).wait()

    assert mock_gretel_trainer.called_with["train"] == (csv, ",")
    assert mock_gretel_trainer.called_with["generate"] == (3,)
    assert mock_gretel_trainer.called_with["get_sqs_score"] == ()

    factory_args = mock_gretel_trainer.called_with["_factory_args"]
    assert factory_args["project_name"] == "benchmark-proj-0"
    assert (
        factory_args["cache_file"]
        == f"{TEST_BENCHMARK_DIR}/benchmark-proj-0-runner.json"
    )
    if expected_model_type is None:
        assert factory_args["model_type"] is None
    else:
        assert isinstance(factory_args["model_type"], expected_model_type)


def test_gptx_uses_sdk_executor(csv):
    mock_trainer_factory = Mock()

    mock_record_handler = Mock()
    mock_record_handler.get_artifact_link = Mock()
    mock_model = Mock()
    mock_model.create_record_handler_obj = Mock(return_value=mock_record_handler)
    mock_model.peek_report = Mock(
        return_value={"synthetic_data_quality_score": {"score": 94}}
    )
    mock_project = Mock()
    mock_project.create_model_obj = Mock(return_value=mock_model)
    mock_project_factory = Mock(return_value=mock_project)
    mock_poll = Mock()

    language_dataframe = pd.DataFrame(
        {"text": ["hello world", "hola mundo", "bonjour monde"]}
    )
    language_dataset = _make_dataset(
        [language_dataframe], datatype=Datatype.NATURAL_LANGUAGE
    )

    with patch("pandas.read_csv") as patched_read_csv:
        patched_read_csv.return_value = pd.DataFrame()
        comparison = compare(
            datasets=[language_dataset],
            models=[GretelGPTX],
            runtime_config=TEST_RUNTIME_CONFIG,
            gretel_sdk=_make_gretel_sdk(
                create_project=mock_project_factory,
                poll=mock_poll,
            ),
            gretel_trainer_factory=mock_trainer_factory,
        ).wait()

    mock_trainer_factory.assert_not_called()
    mock_project_factory.assert_called_with("benchmark-proj-0")
    mock_model.submit_cloud.assert_called_once()
    mock_record_handler.submit_cloud.assert_called_once()
    mock_record_handler.get_artifact_link.assert_called_once_with("data")
    assert mock_poll.call_count == 2
    assert comparison.results["Status"].values.tolist() == ["Completed"]
    assert comparison.results["SQS"].values.tolist() == [94]


def test_gretel_model_with_bad_custom_config_fails_before_execution_starts(csv):
    class BadGretelConfig(GretelModel):
        config = {"hello": "world", "hola": "mundo"}

    mock_trainer_factory = Mock()
    mock_project_factory = Mock()
    csv_dataset = _make_dataset([csv])

    with pytest.raises(BenchmarkException):
        compare(
            datasets=[csv_dataset],
            models=[
                BadGretelConfig,
            ],
            runtime_config=TEST_RUNTIME_CONFIG,
            gretel_sdk=_make_gretel_sdk(create_project=mock_project_factory),
            gretel_trainer_factory=mock_trainer_factory,
        )

    mock_trainer_factory.assert_not_called()
    mock_project_factory.assert_not_called()


def test_list_gretel_datasets_and_tags():
    tags = TEST_GRETEL_DATASET_REPO.list_tags()
    assert len(tags) > 0
    for expected_tag in ["small", "large", "e-commerce", "healthcare"]:
        assert expected_tag in tags

    assert (
        len(TEST_GRETEL_DATASET_REPO.list_datasets(datatype=Datatype.TABULAR_MIXED)) > 0
    )

    assert len(TEST_GRETEL_DATASET_REPO.list_datasets(tags=["large"])) > 0

    assert len(TEST_GRETEL_DATASET_REPO.list_datasets(tags=["large", "small"])) == 0

    assert len(TEST_GRETEL_DATASET_REPO.list_datasets(tags=[str(uuid.uuid4())])) == 0

    with pytest.raises(BenchmarkException):
        TEST_GRETEL_DATASET_REPO.get_dataset(str(uuid.uuid4()))


def test_run_comparison_with_gretel_dataset():
    with suppress(FileNotFoundError):
        shutil.rmtree(TEST_BENCHMARK_DIR)

    iris = TEST_GRETEL_DATASET_REPO.get_dataset("iris")

    comparison = compare(
        datasets=[iris],
        models=[
            GretelAuto,
        ],
        runtime_config=TEST_RUNTIME_CONFIG,
        gretel_sdk=_make_gretel_sdk(),
        gretel_trainer_factory=mock_gretel_trainer_factory(get_sqs_score=84),
    ).wait()

    assert comparison.results["Input data"].values.tolist() == ["iris/data.csv"]
    assert comparison.results["Status"].values.tolist() == ["Completed"]

    with suppress(FileNotFoundError):
        assert len(os.listdir(TEST_BENCHMARK_DIR)) == 0
        shutil.rmtree(TEST_BENCHMARK_DIR)


def test_benchmark_cleans_up_after_failures(csv):
    mock_project = Mock()
    mock_search_projects = Mock(return_value=[mock_project])

    csv_dataset = _make_dataset([csv])

    comparison = compare(
        datasets=[csv_dataset],
        models=[
            GretelAuto,
        ],
        runtime_config=TEST_RUNTIME_CONFIG,
        gretel_sdk=_make_gretel_sdk(search_projects=mock_search_projects),
        gretel_trainer_factory=mock_gretel_trainer_factory(fail="train"),
    ).wait()

    assert comparison.results["Status"].values.tolist() == ["Failed (train)"]

    mock_project.delete.assert_called()


def test_make_dataset_with_different_datatypes(csv):
    _make_dataset([csv], datatype=Datatype.TABULAR_MIXED)
    _make_dataset([csv], datatype="tabular_mixed")
    with pytest.raises(BenchmarkException):
        _make_dataset([csv], datatype="invalid datatype")


def test_runs_with_gptx_are_skipped_when_too_many_columns_or_wrong_datatype():
    too_many_columns = _make_dataset(
        [
            pd.DataFrame(
                {
                    "one": ["hello", "hola", "bonjour"],
                    "two": ["world", "mundo", "monde"],
                }
            )
        ],
        datatype=Datatype.NATURAL_LANGUAGE,
    )
    wrong_datatype = _make_dataset(
        [
            pd.DataFrame(
                {
                    "one": ["hello", "hola", "bonjour"],
                }
            )
        ]
    )

    comparison = compare(
        datasets=[too_many_columns, wrong_datatype],
        models=[GretelGPTX],
        runtime_config=TEST_RUNTIME_CONFIG,
        gretel_sdk=_make_gretel_sdk(),
        gretel_trainer_factory=Mock(),
    ).wait()

    assert comparison.results["Status"].values.tolist() == ["Skipped", "Skipped"]


def test_runs_with_lstm_are_skipped_when_over_150_columns():
    too_many_columns = _make_dataset(
        [pd.DataFrame(index=range(151), columns=range(151))]
    )

    comparison = compare(
        datasets=[too_many_columns],
        models=[GretelLSTM],
        runtime_config=TEST_RUNTIME_CONFIG,
        gretel_sdk=_make_gretel_sdk(),
        gretel_trainer_factory=Mock(),
    ).wait()

    assert comparison.results["Status"].values.tolist() == ["Skipped"]


def test_skip_cleanup_when_requested():
    with suppress(FileNotFoundError):
        shutil.rmtree(TEST_BENCHMARK_DIR)

    runtime_config = TEST_RUNTIME_CONFIG
    runtime_config.auto_clean = False

    mock_project = Mock()
    mock_search_projects = Mock(return_value=[mock_project])

    dataset = TEST_GRETEL_DATASET_REPO.list_datasets()[0]

    compare(
        datasets=[dataset],
        models=[GretelLSTM],
        runtime_config=runtime_config,
        gretel_sdk=_make_gretel_sdk(search_projects=mock_search_projects),
        gretel_trainer_factory=lambda **kw: MockGretelTrainer(),
    ).wait()

    assert not mock_project.delete.called
    assert len(os.listdir(TEST_BENCHMARK_DIR)) > 0
    shutil.rmtree(TEST_BENCHMARK_DIR)


def test_exits_early_when_session_is_misconfigured(csv):
    def mock_configure_session():
        raise Exception("invalid creds")

    mock_trainer_factory = Mock()
    mock_project_factory = Mock()
    csv_dataset = _make_dataset([csv])

    with pytest.raises(Exception):
        compare(
            datasets=[csv_dataset],
            models=[GretelLSTM],
            runtime_config=TEST_RUNTIME_CONFIG,
            gretel_sdk=_make_gretel_sdk(
                configure_session=mock_configure_session,
                create_project=mock_project_factory,
            ),
            gretel_trainer_factory=mock_trainer_factory,
        )

    mock_trainer_factory.assert_not_called()
    mock_project_factory.assert_not_called()
