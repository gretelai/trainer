from dataclasses import dataclass
from typing import Callable, Dict, List
from typing_extensions import Protocol

import pandas as pd

from gretel_trainer.benchmark.core import DataSource, Datatype, Evaluator
from gretel_trainer.benchmark.gretel.models import GretelModel, GretelModelConfig

import gretel_client.helpers

from gretel_client.evaluation.quality_report import QualityReport
from gretel_client.projects.projects import create_or_get_unique_project, search_projects


class GretelSDKJob(Protocol):
    def submit_cloud(self) -> None:
        ...


class GretelSDKRecordHandler(GretelSDKJob, Protocol):
    def get_artifact_link(self, artifact_key: str) -> str:
        ...


class GretelSDKModel(GretelSDKJob, Protocol):
    def create_record_handler_obj(self, params: Dict) -> GretelSDKRecordHandler:
        ...

    def peek_report(self) -> Dict:
        ...


class GretelSDKProject(Protocol):
    def create_model_obj(
        self, model_config: GretelModelConfig, data_source: str
    ) -> GretelSDKModel:
        ...

    def delete(self) -> None:
        ...


Poll = Callable[[GretelSDKJob], None]


@dataclass
class GretelSDK:
    create_project: Callable[..., GretelSDKProject]
    search_projects: Callable[..., List[GretelSDKProject]]
    evaluate: Evaluator
    poll: Poll


def _create_project(name: str) -> GretelSDKProject:
    return create_or_get_unique_project(name=name)


def _search_projects(query: str) -> List[GretelSDKProject]:
    return search_projects(query=query)


def _evaluate(synthetic: pd.DataFrame, reference: str) -> int:
    report = QualityReport(data_source=synthetic, ref_data=reference)
    report.run()
    return report.peek()["score"]


ActualGretelSDK = GretelSDK(
    create_project=_create_project,
    search_projects=_search_projects,
    evaluate=_evaluate,
    poll=gretel_client.helpers.poll,
)


class GretelSDKExecutor:
    def __init__(
        self,
        project_name: str,
        model: GretelModel,
        model_key: str,
        sdk: GretelSDK,
    ):
        self.project_name = project_name
        self.model = model
        self.model_key = model_key
        self.sdk = sdk

    @property
    def model_name(self) -> str:
        return self.model.name

    def runnable(self, source: DataSource) -> bool:
        if self.model_key == "gpt_x":
            if source.column_count > 1 or source.datatype != Datatype.NATURAL_LANGUAGE:
                return False

        return True

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
