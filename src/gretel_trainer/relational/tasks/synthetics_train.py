from typing import Dict, List

from gretel_client.projects.jobs import Job
from gretel_client.projects.models import Model
from gretel_client.projects.projects import Project

from gretel_trainer.relational.tasks.common import _MultiTable
from gretel_trainer.relational.workflow_state import SyntheticsTrain


class SyntheticsTrainTask:
    def __init__(
        self,
        synthetics_train: SyntheticsTrain,
        multitable: _MultiTable,
    ):
        self.synthetics_train = synthetics_train
        self.multitable = multitable
        self.completed = []
        self.failed = []

    @property
    def action(self) -> str:
        return "synthetics model training"

    @property
    def refresh_interval(self) -> int:
        return self.multitable._refresh_interval

    @property
    def project(self) -> Project:
        return self.multitable._project

    @property
    def table_collection(self) -> List[str]:
        return list(self.synthetics_train.models.keys())

    @property
    def artifacts_per_job(self) -> int:
        return 1

    def more_to_do(self) -> bool:
        return len(self.completed + self.failed) < len(self.synthetics_train.models)

    def is_finished(self, table: str) -> bool:
        return table in (self.completed + self.failed)

    def get_job(self, table: str) -> Job:
        return self.synthetics_train.models[table]

    def handle_completed(self, table: str, job: Job) -> None:
        self.completed.append(table)

    def handle_failed(self, table: str) -> None:
        self.failed.append(table)

    def handle_lost_contact(self, table: str) -> None:
        self.synthetics_train.lost_contact.append(table)
        self.failed.append(table)

    def each_iteration(self) -> None:
        self.multitable._backup()
