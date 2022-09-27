import os

from contextlib import suppress
from pathlib import Path
from typing import Callable, Optional, Protocol

import gretel_trainer
import pandas as pd

from gretel_trainer.benchmark.core import DataSource
from gretel_trainer.benchmark.gretel.models import GretelModel, GretelModelConfig
from gretel_trainer import models

from gretel_client.projects.models import read_model_config


def _get_trainer_model_type(
    config: GretelModelConfig,
) -> Optional[gretel_trainer.models._BaseConfig]:
    if config == "AUTO":
        return None

    config_dict = read_model_config(config)
    model_name = list(config_dict["models"][0])[0]

    if model_name == "synthetics":
        model_class = models.GretelLSTM
    elif model_name == "ctgan":
        model_class = models.GretelCTGAN
    else:
        raise Exception(f"Unexpected model name 'f{model_name}' in config")

    return model_class(config=config_dict)


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
        model: GretelModel,
        trainer_factory: Callable[..., Trainer],
        delete_project: Callable[[str], None],
        benchmark_dir: str,
    ):
        self.project_name = project_name
        self.model = model
        self.trainer_factory = trainer_factory
        self.delete_project = delete_project
        self.benchmark_dir = benchmark_dir
        self.trainer_cache_file = f"{benchmark_dir}/{self.project_name}-runner.json"

    @property
    def model_name(self) -> str:
        return self.model.name

    def runnable(self, source: DataSource) -> bool:
        return True

    def train(self, source: str, **kwargs) -> None:
        Path(self.benchmark_dir).mkdir(exist_ok=True)
        self.trainer_model = self.trainer_factory(
            project_name=self.project_name,
            model_type=_get_trainer_model_type(self.model.config),
            cache_file=self.trainer_cache_file,
        )
        self.trainer_model.train(source, delimiter=kwargs["delimiter"])

    def generate(self, **kwargs) -> pd.DataFrame:
        return self.trainer_model.generate(num_records=kwargs["training_row_count"])

    def get_sqs_score(self, synthetic: pd.DataFrame, reference: str) -> int:
        return self.trainer_model.get_sqs_score()

    def cleanup(self) -> None:
        self.delete_project(self.project_name)
        with suppress(FileNotFoundError):
            os.remove(self.trainer_cache_file)
