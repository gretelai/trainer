from dataclasses import dataclass, field
from typing import Optional, Union

import pandas as pd

from gretel_trainer.b2.core import BenchmarkException, DataSourceTypes, Datatype


@dataclass
class CustomDataset:
    source: Union[str, pd.DataFrame] = field(repr=False)
    datatype: Datatype
    name: str
    row_count: int = field(init=False)
    column_count: int = field(init=False)

    def __post_init__(self):
        if isinstance(self.source, str):
            df = pd.read_csv(self.source)
        else:
            df = self.source

        self.row_count = df.shape[0]
        self.column_count = df.shape[1]

    @property
    def data_source(self) -> Union[str, pd.DataFrame]:
        return self.source


def _make_dataset_from_str(
    source: str,
    datatype: Datatype,
    name: Optional[str]
) -> CustomDataset:
    return CustomDataset(
        source=source,
        datatype=datatype,
        name=name or source
    )


def _make_dataset_from_dataframe(
    source: pd.DataFrame,
    datatype: Datatype,
    name: Optional[str]
) -> CustomDataset:
    if name is None:
        raise BenchmarkException("`name` is required for datasets created from Pandas DataFrames")
    return CustomDataset(
        source=source,
        datatype=datatype,
        name=name,
    )


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
    name: Optional[str] = None
) -> CustomDataset:
    datatype = _to_datatype(datatype)
    if isinstance(source, str):
        return _make_dataset_from_str(source, datatype, name)
    elif isinstance(source, pd.DataFrame):
        return _make_dataset_from_dataframe(source, datatype, name)
    else:
        raise BenchmarkException("`source` must be either a string path to a CSV or a Pandas DataFrame")
