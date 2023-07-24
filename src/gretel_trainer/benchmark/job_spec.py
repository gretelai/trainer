from dataclasses import dataclass
from typing import Union

from gretel_trainer.benchmark.core import Dataset
from gretel_trainer.benchmark.custom.models import CustomModel
from gretel_trainer.benchmark.gretel.models import GretelModel


# TODO(pm): Move this and JobSpec to `core`?
class RunKey(tuple[str, str]):
    RUN_IDENTIFIER_SEPARATOR = "-"

    @property
    def model_name(self) -> str:
        return self[0]

    @property
    def dataset_name(self) -> str:
        return self[1]

    @property
    def identifier(self) -> str:
        """Identifier returns a flat string identifying this run.

        Returns:
            A string identifying the run.
        """
        return self.RUN_IDENTIFIER_SEPARATOR.join(self)

    def __repr__(self) -> str:
        return f"({self.model_name}, {self.dataset_name})"


@dataclass
class JobSpec:
    # TODO(pm): for simplicity just use Dataset instead of union
    dataset: Dataset

    # TODO(pm): for simplicity, we only allow storing instantiated models here, no Type[X]
    model: Union[GretelModel, CustomModel]

    def make_run_key(self):
        return RunKey((model_name(self.model), self.dataset.name))


def model_name(model: Union[GretelModel, CustomModel]) -> str:
    if isinstance(model, GretelModel):
        return model.name
    else:
        return type(model).__name__
