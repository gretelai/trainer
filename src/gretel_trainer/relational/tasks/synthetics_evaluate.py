from typing import Dict, List

from gretel_client.projects.jobs import Job
from gretel_client.projects.models import Model
from gretel_client.projects.projects import Project

from gretel_trainer.relational.tasks.common import _MultiTable


class SyntheticsEvaluateTask:
    def __init__(
        self,
        evaluate_models: Dict[str, Model],
        project: Project,
        multitable: _MultiTable,
    ):
        self.evaluate_models = evaluate_models
        self.project = project
        self.multitable = multitable
        self.completed = []
        self.failed = []

    @property
    def action(self) -> str:
        return "synthetic data evaluation"

    @property
    def refresh_interval(self) -> int:
        return 20

    @property
    def table_collection(self) -> List[str]:
        return list(self.evaluate_models.keys())

    @property
    def artifacts_per_job(self) -> int:
        return 2

    def more_to_do(self) -> bool:
        return len(self.completed + self.failed) < len(self.evaluate_models)

    def is_finished(self, table: str) -> bool:
        return table in (self.completed + self.failed)

    def get_job(self, table: str) -> Job:
        return self.evaluate_models[table]

    def handle_completed(self, table: str, job: Job) -> None:
        self.completed.append(table)

    def handle_failed(self, table: str) -> None:
        self.failed.append(table)

    def handle_lost_contact(self, table: str) -> None:
        self.failed.append(table)

    def each_iteration(self) -> None:
        pass
