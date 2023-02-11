from multiprocessing.managers import DictProxy

from gretel_trainer.b2.core import Dataset, RunIdentifier
from gretel_trainer.b2.gretel_models import GretelModel
from gretel_trainer.b2.status import NotStarted, RunStatus


class GretelTrainerExecutor:
    def __init__(
        self,
        project_prefix: str,
        benchmark_model: GretelModel,
        run_identifier: RunIdentifier,
        statuses: DictProxy,
    ):
        self.project_prefix = project_prefix
        self.benchmark_model = benchmark_model
        self.run_identifier = run_identifier
        self.statuses = statuses
        self.set_status(NotStarted())

    def set_status(self, status: RunStatus) -> None:
        self.status = status
        self.statuses[self.run_identifier] = status

    def train(self, dataset: Dataset) -> None:
        pass

    def generate(self) -> None:
        pass
