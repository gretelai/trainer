import pandas as pd

from gretel_trainer.b2.core import Dataset


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
