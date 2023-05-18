from gretel_client.projects.jobs import Job
from gretel_client.projects.models import Model
from gretel_client.projects.projects import Project

import gretel_trainer.relational.tasks.common as common

ACTION = "synthetic data evaluation"


class SyntheticsEvaluateTask:
    def __init__(
        self,
        evaluate_models: dict[str, Model],
        project: Project,
        multitable: common._MultiTable,
    ):
        self.evaluate_models = evaluate_models
        self.project = project
        self.multitable = multitable
        self.completed = []
        self.failed = []

    def action(self, job: Job) -> str:
        return ACTION

    @property
    def table_collection(self) -> list[str]:
        return list(self.evaluate_models.keys())

    @property
    def artifacts_per_job(self) -> int:
        return 2

    def more_to_do(self) -> bool:
        return len(self.completed + self.failed) < len(self.evaluate_models)

    def wait(self) -> None:
        common.wait(20)

    def is_finished(self, table: str) -> bool:
        return table in (self.completed + self.failed)

    def get_job(self, table: str) -> Job:
        return self.evaluate_models[table]

    def handle_completed(self, table: str, job: Job) -> None:
        self.completed.append(table)
        common.log_success(table, ACTION)
        common.cleanup(sdk=self.multitable._extended_sdk, project=self.project, job=job)

    def handle_failed(self, table: str, job: Job) -> None:
        self.failed.append(table)
        common.log_failed(table, ACTION)
        common.cleanup(sdk=self.multitable._extended_sdk, project=self.project, job=job)

    def handle_lost_contact(self, table: str, job: Job) -> None:
        self.failed.append(table)
        common.log_lost_contact(table)
        common.cleanup(sdk=self.multitable._extended_sdk, project=self.project, job=job)

    def handle_in_progress(self, table: str, job: Job) -> None:
        common.log_in_progress(table, job.status, ACTION)

    def each_iteration(self) -> None:
        pass
