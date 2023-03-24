import json
import tempfile
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from gretel_trainer.b2 import GretelDatasetRepo

REPO = GretelDatasetRepo()
IRIS = REPO.get_dataset("iris")


@pytest.fixture(autouse=True)
def patch_configure_session():
    with patch("gretel_trainer.b2.comparison.configure_session"):
        yield


@pytest.fixture(autouse=True)
def sleepless():
    with patch("time.sleep"):
        yield


@pytest.fixture()
def iris():
    return IRIS


@pytest.fixture()
def df():
    return pd.DataFrame(data={"name": ["Tapu", "Rolo", "Archie"], "age": [6, 14, 8]})


@pytest.fixture()
def project():
    with patch("gretel_trainer.b2.comparison.create_project") as create_project:
        project = Mock()
        create_project.return_value = project
        yield project


@pytest.fixture()
def evaluate_report_path():
    report = {"synthetic_data_quality_score": {"score": 95}}
    with tempfile.NamedTemporaryFile() as f:
        with open(f.name, "w") as j:
            json.dump(report, j)
        yield f.name


@pytest.fixture()
def working_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
