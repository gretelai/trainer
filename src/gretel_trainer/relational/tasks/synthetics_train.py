from typing import Dict, List

from gretel_client.projects.jobs import Job
from gretel_client.projects.models import Model
from gretel_client.projects.projects import Project

from gretel_trainer.relational.tasks.common import _MultiTable


class SyntheticsTrainTask:
    def __init__(
        self,
        models: Dict[str, Model],
        lost_contact: List[str],
        multitable: _MultiTable,
    ):
        self.models = models
        self.lost_contact = lost_contact
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
        return list(self.models.keys())

    @property
    def artifacts_per_job(self) -> int:
        return 1

    def more_to_do(self) -> bool:
        return len(self.completed + self.failed) < len(self.models)

    def is_finished(self, table: str) -> bool:
        return table in (self.completed + self.failed)

    def get_job(self, table: str) -> Job:
        return self.models[table]

    def handle_completed(self, table: str, job: Job) -> None:
        self.completed.append(table)
        # TODO strategy update evaluation from model???
        # Maybe for simplicity do this outside the task loop

    def handle_failed(self, table: str) -> None:
        self.failed.append(table)

    def handle_lost_contact(self, table: str) -> None:
        self.lost_contact.append(table)
        self.failed.append(table)

    def each_iteration(self) -> None:
        self.multitable._backup()
