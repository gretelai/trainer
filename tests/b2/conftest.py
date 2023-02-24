from unittest.mock import patch

import pytest

from gretel_trainer.b2 import GretelDatasetRepo

REPO = GretelDatasetRepo()
IRIS = REPO.get_dataset("iris")


@pytest.fixture()
def nosleep():
    with patch("time.sleep"):
        yield


@pytest.fixture()
def iris():
    return IRIS
