from pathlib import Path
from typing import Callable, Optional
from typing_extensions import Protocol

import gretel_trainer
import pandas as pd


class Trainer(Protocol):
    def train(self, source: str, delimiter: Optional[str]) -> None:
        ...

    def generate(self, num_records: int) -> pd.DataFrame:
        ...

    def get_sqs_score(self) -> int:
        ...


class GretelTrainerExecutor:
    def __init__(
        self,
        project_name: str,
        trainer_model_type: Optional[gretel_trainer.models._BaseConfig],
        trainer_factory: Callable[..., Trainer],
        benchmark_dir: str,
    ):
        self.project_name = project_name
        self.trainer_model_type = trainer_model_type
        self.trainer_factory = trainer_factory
        self.benchmark_dir = benchmark_dir

    def train(self, source: str, **kwargs) -> None:
        Path(self.benchmark_dir).mkdir(exist_ok=True)
        cache_file = f"{self.benchmark_dir}/{self.project_name}-runner.json"
        self.trainer_model = self.trainer_factory(
            project_name=self.project_name,
            model_type=self.trainer_model_type,
            cache_file=cache_file,
        )
        self.trainer_model.train(source, delimiter=kwargs["delimiter"])

    def generate(self, **kwargs) -> pd.DataFrame:
        return self.trainer_model.generate(num_records=kwargs["training_row_count"])

    def get_sqs_score(self, synthetic: pd.DataFrame, reference: str) -> int:
        return self.trainer_model.get_sqs_score()
