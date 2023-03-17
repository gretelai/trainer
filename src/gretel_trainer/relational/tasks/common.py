from typing_extensions import Protocol


class _MultiTable(Protocol):
    @property
    def _refresh_interval(self) -> int:
        ...

    def _backup(self) -> None:
        ...
