"""
TODO
"""
from __future__ import annotations
from typing import TYPE_CHECKING
from pathlib import Path

from dataclasses import dataclass


if TYPE_CHECKING:
    from gretel_trainer.relational.connectors import Connector
    from gretel_trainer.relational.core import RelationalData


@dataclass
class SubsetConfig:
    target_row_count: float
    """
    The target number of rows to sample. This will be used as the sample target for "leaf" tables, 
    or tables that do not have any references to their primary keys. If this number is > 1 then
    that number of rows will be used, if the value is <= 1 then it is considered to be a percetange
    of the total number of rows.
    """

    def __post_init__(self):
        if self.target_row_count <= 0:
            raise ValueError("The `row_count` must be greather than 0.")

    def calculate_row_count(self, row_count: int) -> int:
        if self.target_row_count > 1:
            return row_count

        return int(row_count * self.target_row_count)


class Subsetter:
    _connector: Connector
    _rel_data: RelationalData
    _config: SubsetConfig
    _storage_dir: Path

    def __init__(
        self,
        *,
        config: SubsetConfig,
        connector: Connector,
        relational_data: RelationalData,
        storage_dir: Path,
    ):
        self._connector = connector
        self._rel_data = relational_data
        self._config = config

        if not storage_dir.is_dir():
            raise ValueError("The `storage_dir` must be a directory!")

        self._storage_dir = storage_dir
