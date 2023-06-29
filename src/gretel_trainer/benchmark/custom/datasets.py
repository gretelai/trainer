import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from gretel_trainer.benchmark.core import BenchmarkException, Datatype, get_data_shape

logger = logging.getLogger(__name__)


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


def create_dataset(
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


def make_dataset(
    sources: Union[list[str], list[pd.DataFrame]],
    *,
    datatype: Union[str, Datatype],
    namespace: Optional[str] = None,
    delimiter: str = ",",
) -> CustomDataset:
    logger.warning(
        "`make_dataset` is deprecated and will be removed in a future release. Please use `create_dataset` instead."
    )

    if not isinstance(sources, list):
        raise BenchmarkException(
            "Did not receive list argument to `sources`, but instead of adjusting, please use `create_dataset` instead of this deprecated function."
        )

    if len(sources) > 1:
        raise BenchmarkException(
            "`make_dataset` no longer supports multiple sources. Please create separate datasets using `create_dataset`."
        )

    source = sources[0]

    if isinstance(source, pd.DataFrame):
        ns = namespace or "DataFrames"
        shorthash = str(uuid.uuid4())[:8]
        name = f"{ns}::{shorthash}"
    else:
        ns = f"{namespace}::" if namespace else ""
        name = f"{ns}{source}"

    return create_dataset(
        source=source,
        datatype=datatype,
        name=name,
        delimiter=delimiter,
    )
