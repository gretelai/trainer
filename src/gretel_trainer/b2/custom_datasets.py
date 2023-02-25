import os
from dataclasses import dataclass, field
from typing import Optional, Union

import pandas as pd

from gretel_trainer.b2.core import BenchmarkException, Datatype


@dataclass
class CustomDataset:
    name: str
    datatype: Datatype
    data_source: Union[str, pd.DataFrame] = field(repr=False)
    row_count: int = field(init=False, repr=False)
    column_count: int = field(init=False, repr=False)

    def __post_init__(self):
        if isinstance(self.data_source, str):
            df = pd.read_csv(self.data_source)
        else:
            df = self.data_source
        self.row_count = df.shape[0]
        self.column_count = df.shape[1]


def _to_datatype(d: Union[str, Datatype]) -> Datatype:
    if isinstance(d, Datatype):
        return d
    try:
        return Datatype[d]
    except KeyError:
        raise BenchmarkException("Unrecognized datatype requested")


def make_dataset(
    source: Union[str, pd.DataFrame],
    *,
    datatype: Union[str, Datatype],
    name: str,
) -> CustomDataset:
    datatype = _to_datatype(datatype)
    if not isinstance(source, (str, pd.DataFrame)):
        raise BenchmarkException(
            "`source` must be either a string path to a CSV or a Pandas DataFrame"
        )
    if isinstance(source, str) and not os.path.isfile(source):
        raise BenchmarkException("String `source` must be a path to a file")

    return CustomDataset(data_source=source, datatype=datatype, name=name)
