import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import pandas as pd

from gretel_trainer.benchmark.core import BenchmarkException, Datatype, get_data_shape


@dataclass
class CustomDataset:
    name: str
    datatype: Datatype
    data_source: Union[str, pd.DataFrame] = field(repr=False)
    row_count: int = field(init=False, repr=False)
    column_count: int = field(init=False, repr=False)
    delimiter: str

    def __post_init__(self):
        if isinstance(self.data_source, str):
            rows, cols = get_data_shape(self.data_source, self.delimiter)
        else:
            rows, cols = self.data_source.shape
        self.row_count = rows
        self.column_count = cols

    @property
    def public(self) -> bool:
        return False  # custom datasets can't be public, currently


def _to_datatype(d: Union[str, Datatype]) -> Datatype:
    if isinstance(d, Datatype):
        return d
    try:
        return Datatype(d.lower())
    except ValueError:
        raise BenchmarkException("Unrecognized datatype requested")


def make_dataset(
    source: Union[str, pd.DataFrame],
    *,
    datatype: Union[str, Datatype],
    name: str,
    delimiter: str = ",",
) -> CustomDataset:
    datatype = _to_datatype(datatype)
    if not isinstance(source, (str, pd.DataFrame)):
        raise BenchmarkException(
            "`source` must be either a string path to a CSV or a Pandas DataFrame"
        )
    if isinstance(source, str) and not Path(os.path.expanduser(source)).exists():
        raise BenchmarkException("String `source` must be a path to a file")

    return CustomDataset(
        data_source=source, datatype=datatype, name=name, delimiter=delimiter
    )
