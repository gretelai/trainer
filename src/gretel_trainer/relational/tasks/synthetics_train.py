import gretel_trainer.relational.tasks.common as common

from gretel_client.projects.jobs import Job
from gretel_trainer.relational.task_runner import TaskContext
from gretel_trainer.relational.workflow_state import SyntheticsTrain

ACTION = "synthetics model training"


class SyntheticsTrainTask:
    def __init__(
        self,
        synthetics_train: SyntheticsTrain,
        ctx: TaskContext,
    ):
        self.synthetics_train = synthetics_train
        self.ctx = ctx
        self.completed = []
        self.failed = []

    def action(self, job: Job) -> str:
        return ACTION

    @property
    def table_collection(self) -> list[str]:
        return list(self.synthetics_train.models.keys())

    def more_to_do(self) -> bool:
        return len(self.completed + self.failed) < len(self.synthetics_train.models)

    def is_finished(self, table: str) -> bool:
        return table in (self.completed + self.failed)

    def get_job(self, table: str) -> Job:
        return self.synthetics_train.models[table]

    def handle_completed(self, table: str, job: Job) -> None:
        self.completed.append(table)
        common.log_success(table, self.action(job))
        common.cleanup(sdk=self.ctx.extended_sdk, project=self.ctx.project, job=job)

    def handle_failed(self, table: str, job: Job) -> None:
        self.failed.append(table)
        common.log_failed(table, self.action(job))
        common.cleanup(sdk=self.ctx.extended_sdk, project=self.ctx.project, job=job)

    def handle_lost_contact(self, table: str, job: Job) -> None:
        self.synthetics_train.lost_contact.append(table)
        self.failed.append(table)
        common.log_lost_contact(table)
        common.cleanup(sdk=self.ctx.extended_sdk, project=self.ctx.project, job=job)

    def handle_in_progress(self, table: str, job: Job) -> None:
        common.log_in_progress(table, job.status, self.action(job))

    def each_iteration(self) -> None:
        self.ctx.backup()
