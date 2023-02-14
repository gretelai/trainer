from dataclasses import dataclass, field
from typing import Optional, Union

import pandas as pd

from gretel_trainer.b2.core import BenchmarkException, DataSourceTypes, Datatype


@dataclass
class CustomDataset:
    source: Union[str, pd.DataFrame] = field(repr=False)
    datatype: Datatype
    name: str
    df: pd.DataFrame = field(init=False)

    def __post_init__(self):
        if isinstance(self.source, str):
            self.df = pd.read_csv(self.source)
        else:
            self.df = self.source

    @property
    def row_count(self) -> int:
        return self.df.shape[0]

    @property
    def column_count(self) -> int:
        return self.df.shape[1]

    @property
    def data_source(self) -> Union[str, pd.DataFrame]:
        return self.source


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
    return CustomDataset(source=source, datatype=datatype, name=name)
