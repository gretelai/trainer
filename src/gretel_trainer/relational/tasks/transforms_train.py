from gretel_client.projects.jobs import Job
from gretel_client.projects.projects import Project

import gretel_trainer.relational.tasks.common as common
from gretel_trainer.relational.workflow_state import TransformsTrain

ACTION = "transforms model training"


class TransformsTrainTask:
    def __init__(
        self,
        transforms_train: TransformsTrain,
        multitable: common._MultiTable,
    ):
        self.transforms_train = transforms_train
        self.multitable = multitable
        self.completed = []
        self.failed = []

    def action(self, job: Job) -> str:
        return ACTION

    @property
    def project(self) -> Project:
        return self.multitable._project

    @property
    def table_collection(self) -> list[str]:
        return list(self.transforms_train.models.keys())

    @property
    def artifacts_per_job(self) -> int:
        return 1

    def more_to_do(self) -> bool:
        return len(self.completed + self.failed) < len(self.transforms_train.models)

    def wait(self) -> None:
        common.wait(self.multitable._refresh_interval)

    def is_finished(self, table: str) -> bool:
        return table in (self.completed + self.failed)

    def get_job(self, table: str) -> Job:
        return self.transforms_train.models[table]

    def handle_completed(self, table: str, job: Job) -> None:
        self.completed.append(table)
        common.log_success(table, ACTION)
        common.cleanup(sdk=self.multitable._extended_sdk, project=self.project, job=job)

    def handle_failed(self, table: str, job: Job) -> None:
        self.failed.append(table)
        common.log_failed(table, ACTION)
        common.cleanup(sdk=self.multitable._extended_sdk, project=self.project, job=job)

    def handle_lost_contact(self, table: str, job: Job) -> None:
        self.transforms_train.lost_contact.append(table)
        self.failed.append(table)
        common.log_lost_contact(table)
        common.cleanup(sdk=self.multitable._extended_sdk, project=self.project, job=job)

    def handle_in_progress(self, table: str, job: Job) -> None:
        common.log_in_progress(table, job.status, ACTION)

    def each_iteration(self) -> None:
        self.multitable._backup()
