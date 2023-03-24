import tempfile

import pytest

from gretel_trainer.b2 import Datatype, make_dataset
from gretel_trainer.b2.core import BenchmarkException


def test_making_good_datasets(df):
    str_datatype = make_dataset(df, datatype="tabular", name="via_str")
    enum_datatype = make_dataset(df, datatype=Datatype.tabular, name="via_str")
    with tempfile.NamedTemporaryFile() as f:
        df.to_csv(f.name, index=False)
        file_source = make_dataset(f.name, datatype="tabular", name="from_file")

    for dataset in [str_datatype, enum_datatype, file_source]:
        assert dataset.row_count == 3
        assert dataset.column_count == 2


def test_making_bad_datasets(df):
    with pytest.raises(BenchmarkException):
        make_dataset([1, 2, 3], datatype="tabular", name="nope")  # type:ignore

    with pytest.raises(BenchmarkException):
        make_dataset("nonexistent.csv", datatype="tabular", name="nope")

    with pytest.raises(BenchmarkException):
        make_dataset(df, datatype="nonsense", name="nope")
