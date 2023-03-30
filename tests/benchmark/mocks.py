import pandas as pd
from gretel_client.projects.models import read_model_config

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


class SharedDictLstm(GretelModel):
    config = {
        "schema_version": "1.0",
        "name": "tabular-lstm",
        "models": [
            {
                "synthetics": {
                    "data_source": "__tmp__",
                    "params": {
                        "epochs": "auto",
                        "vocab_size": "auto",
                        "learning_rate": "auto",
                        "batch_size": "auto",
                        "rnn_units": "auto",
                    },
                    "generate": {"num_records": 5000},
                    "privacy_filters": {
                        "outliers": "auto",
                        "similarity": "auto",
                    },
                }
            }
        ],
    }
