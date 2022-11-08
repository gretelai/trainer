from dataclasses import dataclass
from typing import Callable, Dict, List
from typing_extensions import Protocol

import pandas as pd

from gretel_trainer.benchmark.core import Evaluator
from gretel_trainer.benchmark.gretel.models import GretelModel, GretelModelConfig

import gretel_client.helpers

from gretel_client import configure_session
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
    configure_session: Callable[[], None]
    create_project: Callable[[str], GretelSDKProject]
    search_projects: Callable[[str], List[GretelSDKProject]]
    evaluate: Evaluator
    poll: Poll


def _configure_session() -> None:
    return configure_session(api_key="prompt", cache="yes", validate=True)


def _create_project(name: str) -> GretelSDKProject:
    return create_or_get_unique_project(name=name)


def _search_projects(query: str) -> List[GretelSDKProject]:
    return search_projects(query=query)


def _evaluate(synthetic: pd.DataFrame, reference: str) -> int:
    report = QualityReport(data_source=synthetic, ref_data=reference)
    report.run()
    return report.peek()["score"]


ActualGretelSDK = GretelSDK(
    configure_session=_configure_session,
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
