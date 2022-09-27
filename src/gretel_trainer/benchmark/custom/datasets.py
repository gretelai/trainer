import csv
import os
import uuid

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Type, Union

import pandas as pd

from gretel_trainer.benchmark.core import BenchmarkException, Datatype

DATA_FRAME_SOURCE_DELIMITER = ","


@dataclass
class DataFrameSource:
    name: str
    datatype: Datatype
    local_dir: str
    df: pd.DataFrame

    @property
    def delimiter(self) -> str:
        return DATA_FRAME_SOURCE_DELIMITER

    @property
    def row_count(self) -> int:
        return self.df.shape[0]

    @property
    def column_count(self) -> int:
        return self.df.shape[1]

    @contextmanager
    def unwrap(self):
        Path(self.local_dir).mkdir(exist_ok=True)
        filename = f"{os.path.expanduser(self.local_dir)}/{uuid.uuid4()}.csv"
        self.df.to_csv(filename, index=False)
        yield filename


@dataclass
class DataFramesDataset:
    dfs: List[pd.DataFrame]
    datatype: Datatype
    local_dir: str
    namespace: str

    def sources(self) -> List[DataFrameSource]:
        return [
            DataFrameSource(
                name=f"{self.namespace}::{index}",
                datatype=self.datatype,
                local_dir=self.local_dir,
                df=df,
            )
            for index, df in enumerate(self.dfs)
        ]


@dataclass
class FileSource:
    name: str
    datatype: Datatype
    path: str
    delimiter: str

    @property
    def row_count(self) -> int:
        with open(os.path.expanduser(self.path)) as f:
            file_rows = sum(1 for _ in f)
        return max(0, file_rows - 1)

    @property
    def column_count(self) -> int:
        with open(os.path.expanduser(self.path)) as f:
            reader = csv.reader(f, delimiter=self.delimiter)
            return len(next(reader))

    @contextmanager
    def unwrap(self):
        yield self.path


@dataclass
class FilesDataset:
    paths: List[str]
    datatype: Datatype
    namespace: Optional[str]
    delimiter: str

    def sources(self) -> List[FileSource]:
        return [
            FileSource(
                name=self._name(path),
                datatype=self.datatype,
                path=path,
                delimiter=self.delimiter,
            )
            for path in self.paths
        ]

    def _name(self, path: str) -> str:
        if self.namespace is None:
            return path
        else:
            return f"{self.namespace}::{path}"


def _validate_dataset_sources(sources: Union[List[str], List[pd.DataFrame]]) -> Type:
    if len(sources) == 0:
        raise BenchmarkException("Dataset must have at least one source")

    first_source_type = type(sources[0])
    if not all(isinstance(source, first_source_type) for source in sources):
        raise BenchmarkException("Dataset sources must all be of the same type")

    if first_source_type not in (str, pd.DataFrame):
        raise BenchmarkException(
            "Dataset sources must be either str (path to CSV file) or pd.DataFrame"
        )

    return first_source_type


def make_dataset(
    sources: Union[List[str], List[pd.DataFrame]],
    *,
    datatype: Union[Datatype, str],
    local_dir: str,
    namespace: Optional[str],
    delimiter: str,
) -> Union[FilesDataset, DataFramesDataset]:
    source_type = _validate_dataset_sources(sources)
    if not isinstance(datatype, Datatype):
        try:
            datatype = Datatype(datatype)
        except ValueError:
            valid_datatypes = [datatype.value for datatype in Datatype]
            raise BenchmarkException(
                f"Invalid datatype: {datatype}. Must be one of: {valid_datatypes}"
            )

    if source_type == pd.DataFrame:
        return DataFramesDataset(
            dfs=sources,
            datatype=datatype,
            local_dir=local_dir,
            namespace=namespace or "DataFrames",
        )
    else:
        return FilesDataset(
            paths=sources, datatype=datatype, delimiter=delimiter, namespace=namespace
        )
