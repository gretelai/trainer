import tempfile

import pandas as pd
import pytest


@pytest.fixture()
def df():
    return pd.DataFrame(data={"name": ["Tapu", "Rolo", "Archie"], "age": [6, 14, 8]})


@pytest.fixture()
def csv(df):
    yield from _tempfile(df, ",")


@pytest.fixture()
def psv(df):
    yield from _tempfile(df, "|")


def _tempfile(dataframe, delimiter):
    with tempfile.NamedTemporaryFile() as f:
        dataframe.to_csv(f.name, sep=delimiter)
        yield f.name
