from typing import Callable, Optional
from typing_extensions import Protocol

import pandas as pd

from gretel_trainer import trainer


class GretelTrainer(Protocol):
    def train(self, source: str, delimiter: Optional[str]) -> None:
        ...

    def generate(self, num_records: int) -> pd.DataFrame:
        ...

    def get_sqs_score(self) -> int:
        ...


TrainerFactory = Callable[..., GretelTrainer]

ActualGretelTrainer: TrainerFactory = trainer.Trainer
