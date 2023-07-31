from typing import Protocol

import pandas as pd

from gretel_trainer.benchmark.core import Dataset


class CustomModel(Protocol):
    def train(self, source: Dataset, **kwargs) -> None:
        ...

    def generate(self, **kwargs) -> pd.DataFrame:
        ...
