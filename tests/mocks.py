from pathlib import Path
from typing import Optional

import pandas as pd

from gretel_trainer.benchmark.gretel.models import GretelModel


class DoNothingModel:
    def train(self, source: str, **kwargs) -> None:
        self.df = pd.read_csv(source)

    def generate(self, **kwargs) -> pd.DataFrame:
        return self.df


class FailingModel:
    @classmethod
    def during_train(cls):
        return cls("train")

    @classmethod
    def during_generate(cls):
        return cls("generate")

    def __init__(self, during: str):
        self.during = during

    def train(self, source: str, **kwargs) -> None:
        self.df = pd.read_csv(source)
        if self.during == "train":
            raise Exception("failed")

    def generate(self, **kwargs) -> pd.DataFrame:
        if self.during == "generate":
            raise Exception("failed")
        return self.df


class LocalFileConfigGretelModel(GretelModel):
    config = f"{Path(__file__).parent}/example_config.yml"


class DictConfigGretelModel(GretelModel):
    config = {
        "schema_version": "1.0",
        "models": [
            {
                "synthetics": {
                    "params": {
                        "epochs": 100,
                        "learning_rate": 0.001,
                    },
                    "validators": {
                        "in_set_count": 10,
                        "pattern_count": 10,
                    },
                    "privacy_filters": {
                        "outliers": "medium",
                        "similarity": "medium",
                    },
                }
            }
        ],
    }


class MockGretelTrainer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.called_with = {}
        self.fail = kwargs.get("fail", None)

    def _factory_args(self, **kwargs):
        self.called_with["_factory_args"] = kwargs
        return self

    def train(self, source: str, delimiter: Optional[str]) -> None:
        self.called_with["train"] = (source, delimiter)
        self._maybe_fail("train")
        return self.kwargs.get("train", None)

    def generate(self, num_records: int) -> pd.DataFrame:
        self.called_with["generate"] = (num_records,)
        self._maybe_fail("generate")
        return self.kwargs.get("generate", None)

    def get_sqs_score(self) -> int:
        self.called_with["get_sqs_score"] = ()
        self._maybe_fail("get_sqs_score")
        return self.kwargs.get("get_sqs_score", None)

    def _maybe_fail(self, key):
        if self.fail == key:
            raise Exception("failed")


def mock_gretel_trainer_factory(**kwargs):
    return lambda **kw: MockGretelTrainer(**kwargs)
