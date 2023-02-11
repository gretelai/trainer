from typing_extensions import Protocol

import pandas as pd

from gretel_trainer.b2.core import Dataset


class CustomModel(Protocol):
    def train(self, source: Dataset, **kwargs) -> None:
        ...

    def generate(self, **kwargs) -> pd.DataFrame:
        ...