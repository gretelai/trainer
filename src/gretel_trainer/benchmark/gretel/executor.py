from typing import Callable, Union

import pandas as pd

from gretel_trainer.benchmark.core import DataSource
from gretel_trainer.benchmark.gretel.models import GretelModel
from gretel_trainer.benchmark.gretel.sdk import GretelSDK, GretelSDKExecutor
from gretel_trainer.benchmark.gretel.trainer import GretelTrainerExecutor, Trainer


class GretelExecutor:
    def __init__(
        self,
        project_name: str,
        model: GretelModel,
        sdk: GretelSDK,
        trainer_factory: Callable[..., Trainer],
        benchmark_dir: str,
    ):
        self._project_name = project_name
        self._model = model
        self._internal_executor = _choose_executor(project_name, model, sdk, trainer_factory, benchmark_dir)

    @property
    def model_name(self) -> str:
        return self._model.name

    def runnable(self, source: DataSource) -> bool:
        return self._model.runnable(source)

    def train(self, source: str, **kwargs) -> None:
        self._internal_executor.train(source, **kwargs)

    def generate(self, **kwargs) -> pd.DataFrame:
        return self._internal_executor.generate(**kwargs)

    def get_sqs_score(self, synthetic: pd.DataFrame, reference: str) -> int:
        return self._internal_executor.get_sqs_score(synthetic, reference)


def _choose_executor(
    project_name: str,
    model: GretelModel,
    sdk: GretelSDK,
    trainer_factory: Callable[..., Trainer],
    benchmark_dir: str,
) -> Union[GretelTrainerExecutor, GretelSDKExecutor]:
    if model.use_trainer:
        return GretelTrainerExecutor(
            project_name,
            model.trainer_model_type,
            trainer_factory,
            benchmark_dir
        )
    else:
        return GretelSDKExecutor(project_name, model, sdk)
