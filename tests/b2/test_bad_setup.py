import os
from unittest.mock import patch

import pytest

from gretel_trainer.b2 import GretelLSTM, compare
from gretel_trainer.b2.core import BenchmarkException


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


def test_dataset_names_must_be_unique(iris):
    with pytest.raises(BenchmarkException):
        compare(
            datasets=[iris, iris],
            models=[GretelLSTM],
            working_dir="should_not_be_created",
        )
    assert not os.path.exists("should_not_be_created")


def test_model_names_must_be_unique(iris):
    with pytest.raises(BenchmarkException):
        compare(
            datasets=[iris],
            models=[GretelLSTM, GretelLSTM],
            working_dir="should_not_be_created",
        )
    assert not os.path.exists("should_not_be_created")
