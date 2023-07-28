from __future__ import annotations

import fnmatch
import math

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from pydantic import BaseModel, Field, PrivateAttr


class RowPartition(BaseModel):
    start: Optional[int]
    end: Optional[int]


class ColumnPartition(BaseModel):
    headers: Optional[List[str]]
    seed_headers: Optional[List[str]]


class Partition(BaseModel):
    idx: int
    rows: RowPartition
    columns: ColumnPartition
    ctx: dict = Field(default_factory=dict)

    def extract_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.columns.headers is not None:
            df = df[self.columns.headers]

        return df.iloc[self.rows.start : self.rows.end]  # noqa

    def update_ctx(self, update: dict):
        self.ctx.update(update)


@dataclass
class PartitionConstraints:
    max_row_count: int
    seed_headers: Optional[List[str]] = None


def _build_partitions(
    df: pd.DataFrame, constraints: PartitionConstraints
) -> List[Partition]:
    total_rows = len(df)

    partitions = []
    partition_idx = 0
    partition_count = math.ceil(total_rows / constraints.max_row_count)

    # We need to break up the number of rows into roughly even chunks
    chunk_size, remain = divmod(total_rows, partition_count)

    # each item in this array is the size of the chunk
    chunks = [chunk_size] * partition_count

    # spread out the remainder evenly across the first N chunks
    for i in range(0, remain):
        chunks[i] += 1

    curr_start = 0
    for chunk_size in chunks:
        seed_headers = constraints.seed_headers
        partitions.append(
            Partition(
                rows=RowPartition(
                    start=curr_start, end=curr_start + chunk_size
                ),
                columns=ColumnPartition(
                    headers=list(df.columns), seed_headers=seed_headers
                ),
                idx=partition_idx,
            )
        )
        partition_idx += 1
        curr_start += chunk_size

    return partitions


class PartitionStrategy(BaseModel):
    id: str
    partitions: List[Partition]
    original_headers: Optional[List[str]]
    status_counter: Optional[dict]
    _disk_location: Path = PrivateAttr(default=None)

    @classmethod
    def from_dataframe(
        cls, id: str, df: pd.DataFrame, constraints: PartitionConstraints
    ) -> PartitionStrategy:
        partitions = _build_partitions(df, constraints)
        return cls(
            id=id,
            partitions=partitions,
            original_headers=list(df.columns),
            status_counter=None,
        )

    @classmethod
    def from_disk(cls, src: Union[Path, str]) -> PartitionStrategy:
        location = Path(src)
        if not location.exists():
            raise ValueError("File does not exist")
        if location.suffix != ".json":
            raise ValueError("Must load from .json")
        instance = cls.parse_file(location)
        # Re-sync to disk to re-set the save location
        instance.save_to(location, overwrite=True)
        return instance

    @property
    def partition_count(self) -> int:
        return len(self.partitions)

    @property
    def row_partition_count(self) -> int:
        return len(self.partitions)

    def save_to(self, dest: Union[Path, str], overwrite: bool = False):
        location = Path(dest)
        if location.suffix != ".json":
            raise ValueError("file must end in .json")
        if location.exists() and not overwrite:
            raise RuntimeError("destination already exists")
        self._disk_location = location
        self.save()

    def save(self):
        if not self._disk_location:
            raise RuntimeError("Save location is unset")
        self._disk_location.write_text(self.json(indent=4))

    def update_partition(self, partition_idx: int, update: dict, autosave: bool = True):
        partition = self.partitions[partition_idx]
        partition.ctx.update(update)
        if autosave:
            self.save()

    def query_partitions(self, query: dict) -> List[Partition]:
        res = []
        if not query:
            return self.partitions
        query_items = query.items()
        for p in self.partitions:
            if not p.ctx:
                continue
            if query_items <= p.ctx.items():
                res.append(p)
        return res

    def query_glob(self, ctx_key: str, query: str) -> List[Partition]:
        res = []
        for p in self.partitions:
            if ctx_key not in p.ctx:
                continue
            if fnmatch.fnmatch(p.ctx[ctx_key], query):
                res.append(p)
        return res

    @property
    def partitions_no_ctx(self) -> List[Partition]:
        return [p for p in self.partitions if not p.ctx]
