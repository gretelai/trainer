from dataclasses import dataclass
from typing import Any, Generic, Optional, TypeVar, Union

from gretel_trainer.benchmark.core import Dataset
from gretel_trainer.benchmark.custom.models import CustomModel
from gretel_trainer.benchmark.gretel.models import GretelModel


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


AnyModelType = Union[GretelModel, CustomModel]
# `covariant=True` is needed, so that `list[GretelModel]` can be used where
# `list[AnyModelType]` is used (see https://peps.python.org/pep-0484/#covariance-and-contravariance)
MODEL_BASE_TYPE = TypeVar("MODEL_BASE_TYPE", bound=AnyModelType, covariant=True)


@dataclass
class JobSpec(Generic[MODEL_BASE_TYPE]):
    dataset: Dataset
    """
    Instance of a dataset that will be used for training.
    """

    model: MODEL_BASE_TYPE
    """
    Instance of a model that will be used.
    """

    metadata: Optional[dict[str, Any]] = None
    """
    Arbitrary metadata that can be attached to the spec for context that may be used
    by some implementation of executors/strategies.
    """

    def make_run_key(self):
        return RunKey((model_name(self.model), self.dataset.name))


def model_name(model: AnyModelType) -> str:
    if isinstance(model, GretelModel):
        return model.name
    else:
        return type(model).__name__
