import tempfile
from unittest.mock import patch

import pandas as pd
import pytest

from gretel_trainer.b2 import GretelDatasetRepo

REPO = GretelDatasetRepo()
IRIS = REPO.get_dataset("iris")


@pytest.fixture()
def iris():
    return IRIS


@pytest.fixture()
def df():
    return pd.DataFrame(data={"name": ["Tapu", "Rolo", "Archie"], "age": [6, 14, 8]})


@pytest.fixture()
def csv(df):
    with tempfile.NamedTemporaryFile() as f:
        df.to_csv(f.name, index=False)
        yield f.name
