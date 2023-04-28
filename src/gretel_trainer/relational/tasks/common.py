from typing import Union

from gretel_client.projects.projects import Project
from typing_extensions import Protocol

from gretel_trainer.relational.core import RelationalData
from gretel_trainer.relational.sdk_extras import ExtendedGretelSDK
from gretel_trainer.relational.strategies.ancestral import AncestralStrategy
from gretel_trainer.relational.strategies.independent import IndependentStrategy


class _MultiTable(Protocol):
    @property
    def _refresh_interval(self) -> int:
        ...

    @property
    def _project(self) -> Project:
        ...

    @property
    def relational_data(self) -> RelationalData:
        ...

    @property
    def _strategy(self) -> Union[AncestralStrategy, IndependentStrategy]:
        ...

    @property
    def _extended_sdk(self) -> ExtendedGretelSDK:
        ...

    def _backup(self) -> None:
        ...
