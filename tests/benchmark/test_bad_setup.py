from unittest.mock import patch

import pytest

from gretel_trainer.benchmark import GretelLSTM, compare
from gretel_trainer.benchmark.core import BenchmarkConfig, BenchmarkException


def test_bad_session_exits_early(iris, tmp_path):
    class SomeException(Exception):
        pass

    should_not_be_created = tmp_path / "should_not_be_created"

    with patch(
        "gretel_trainer.benchmark.entrypoints._verify_client_config"
    ) as verify_client_config:
        verify_client_config.side_effect = SomeException()

        with pytest.raises(SomeException):
            compare(
                datasets=[iris],
                models=[GretelLSTM],
                config=BenchmarkConfig(
                    working_dir=should_not_be_created,
                ),
            )

    assert not should_not_be_created.exists()


def test_dataset_names_must_be_unique(iris, tmp_path):
    should_not_be_created = tmp_path / "should_not_be_created"

    with pytest.raises(BenchmarkException):
        compare(
            datasets=[iris, iris],
            models=[GretelLSTM],
            config=BenchmarkConfig(
                working_dir=should_not_be_created,
            ),
        )
    assert not should_not_be_created.exists()


def test_model_names_must_be_unique(iris, tmp_path):
    should_not_be_created = tmp_path / "should_not_be_created"

    with pytest.raises(BenchmarkException):
        compare(
            datasets=[iris],
            models=[GretelLSTM, GretelLSTM],
            config=BenchmarkConfig(
                working_dir=should_not_be_created,
            ),
        )
    assert not should_not_be_created.exists()
