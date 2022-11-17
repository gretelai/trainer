from dataclasses import dataclass
from typing import Callable, Dict, List
from typing_extensions import Protocol

import pandas as pd

from gretel_trainer.benchmark.core import Evaluator
from gretel_trainer.benchmark.gretel.models import GretelModelConfig

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
