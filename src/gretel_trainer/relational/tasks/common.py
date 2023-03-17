from gretel_client.projects.projects import Project
from typing_extensions import Protocol


class _MultiTable(Protocol):
    @property
    def _refresh_interval(self) -> int:
        ...

    @property
    def _project(self) -> Project:
        ...

    def _backup(self) -> None:
        ...
