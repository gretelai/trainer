import tempfile

import pytest

from gretel_trainer.benchmark import Datatype, create_dataset
from gretel_trainer.benchmark.core import BenchmarkException


def test_creating_good_datasets(df):
    str_datatype = create_dataset(df, datatype="tabular", name="via_str")
    capital_str_datatype = create_dataset(df, datatype="TABULAR", name="via_str")
    enum_datatype = create_dataset(df, datatype=Datatype.TABULAR, name="via_str")
    with tempfile.NamedTemporaryFile() as f:
        df.to_csv(f.name, index=False)
        file_source = create_dataset(f.name, datatype="tabular", name="from_file")

    for dataset in [str_datatype, capital_str_datatype, enum_datatype, file_source]:
        assert dataset.row_count == 3
        assert dataset.column_count == 2


def test_creating_bad_datasets(df):
    with pytest.raises(BenchmarkException):
        create_dataset([1, 2, 3], datatype="tabular", name="nope")  # type:ignore

    with pytest.raises(BenchmarkException):
        create_dataset("nonexistent.csv", datatype="tabular", name="nope")

    with pytest.raises(BenchmarkException):
        create_dataset(df, datatype="nonsense", name="nope")
