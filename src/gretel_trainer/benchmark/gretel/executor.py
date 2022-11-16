from pathlib import Path
from typing import Callable, Optional, Union

import pandas as pd

import gretel_trainer

from gretel_trainer.benchmark.core import DataSource
from gretel_trainer.benchmark.gretel.models import GretelModel
from gretel_trainer.benchmark.gretel.sdk import GretelSDK
from gretel_trainer.benchmark.gretel.trainer import TrainerFactory


class _GretelTrainerExecutor:
    def __init__(
        self,
        project_name: str,
        trainer_model_type: Optional[gretel_trainer.models._BaseConfig],
        trainer_factory: TrainerFactory,
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


class _GretelSDKExecutor:
    def __init__(
        self,
        project_name: str,
        model: GretelModel,
        sdk: GretelSDK,
    ):
        self.project_name = project_name
        self.model = model
        self.sdk = sdk

    def train(self, source: str, **kwargs) -> None:
        project = self.sdk.create_project(self.project_name)
        self.model_obj = project.create_model_obj(
            model_config=self.model.config, data_source=source
        )
        self.model_obj.submit_cloud()
        self.sdk.poll(self.model_obj)

    def generate(self, **kwargs) -> pd.DataFrame:
        record_handler = self.model_obj.create_record_handler_obj(
            params={"num_records": kwargs["training_row_count"]}
        )
        record_handler.submit_cloud()
        self.sdk.poll(record_handler)
        return pd.read_csv(record_handler.get_artifact_link("data"), compression="gzip")

    def get_sqs_score(self, synthetic: pd.DataFrame, reference: str) -> int:
        report = self.model_obj.peek_report()
        if report is None:
            return self.sdk.evaluate(
                synthetic=synthetic, reference=reference
            )
        return report["synthetic_data_quality_score"]["score"]


class GretelExecutor:
    def __init__(
        self,
        project_name: str,
        model: GretelModel,
        sdk: GretelSDK,
        trainer_factory: TrainerFactory,
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
    trainer_factory: TrainerFactory,
    benchmark_dir: str,
) -> Union[_GretelTrainerExecutor, _GretelSDKExecutor]:
    if model.use_trainer:
        return _GretelTrainerExecutor(
            project_name,
            model.trainer_model_type,
            trainer_factory,
            benchmark_dir
        )
    else:
        return _GretelSDKExecutor(project_name, model, sdk)
