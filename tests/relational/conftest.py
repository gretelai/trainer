import itertools
import sqlite3
import tempfile
from pathlib import Path
from typing import Callable, Generator
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from sqlalchemy import create_engine

from gretel_trainer.relational.connectors import Connector
from gretel_trainer.relational.core import RelationalData
from gretel_trainer.relational.sdk_extras import ExtendedGretelSDK

EXAMPLE_DBS = Path(__file__).parent.resolve() / "example_dbs"


@pytest.fixture()
def extended_sdk():
    return ExtendedGretelSDK(hybrid=False)


@pytest.fixture(autouse=True)
def static_suffix(request):
    if "no_mock_suffix" in request.keywords:
        yield
        return
    with patch("gretel_trainer.relational.json.make_suffix") as make_suffix:
        # Each call to make_suffix must be unique or there will be table collisions
        make_suffix.side_effect = itertools.count(start=1)
        yield make_suffix


@pytest.fixture()
def get_invented_table_suffix() -> Callable[[int], str]:
    return _get_invented_table_suffix


def _get_invented_table_suffix(make_suffix_execution_number: int):
    return f"invented_{str(make_suffix_execution_number)}"


@pytest.fixture()
def project():
    with patch(
        "gretel_trainer.relational.multi_table.create_project"
    ) as create_project, patch(
        "gretel_trainer.relational.multi_table.get_project"
    ) as get_project:
        project = Mock()
        project.name = "name"
        project.display_name = "display_name"

        create_project.return_value = project
        get_project.return_value = project

        yield project


def _rel_data_connector(name) -> Connector:
    con = sqlite3.connect(f"file:{name}?mode=memory&cache=shared", uri=True)
    cur = con.cursor()
    with open(EXAMPLE_DBS / f"{name}.sql") as f:
        cur.executescript(f.read())
    return Connector(
        create_engine(f"sqlite:///file:{name}?mode=memory&cache=shared&uri=true")
    )


@pytest.fixture()
def example_dbs():
    return EXAMPLE_DBS


@pytest.fixture()
def tmpdir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture()
def tmpfile():
    with tempfile.NamedTemporaryFile() as tmpfile:
        yield tmpfile


@pytest.fixture()
def pets(tmpdir) -> Generator[RelationalData, None, None]:
    yield _rel_data_connector("pets").extract(storage_dir=tmpdir)


@pytest.fixture()
def ecom(tmpdir) -> Generator[RelationalData, None, None]:
    yield _rel_data_connector("ecom").extract(storage_dir=tmpdir)


@pytest.fixture()
def mutagenesis(tmpdir) -> Generator[RelationalData, None, None]:
    yield _rel_data_connector("mutagenesis").extract(storage_dir=tmpdir)


@pytest.fixture()
def tpch(tmpdir) -> Generator[RelationalData, None, None]:
    yield _rel_data_connector("tpch").extract(storage_dir=tmpdir)


@pytest.fixture()
def art(tmpdir) -> Generator[RelationalData, None, None]:
    yield _rel_data_connector("art").extract(storage_dir=tmpdir)


@pytest.fixture()
def documents(tmpdir) -> Generator[RelationalData, None, None]:
    yield _rel_data_connector("documents").extract(storage_dir=tmpdir)


@pytest.fixture()
def trips(tmpdir) -> Generator[RelationalData, None, None]:
    with tempfile.NamedTemporaryFile() as tmpfile:
        data = pd.DataFrame(
            data={
                "id": list(range(100)),
                "purpose": ["work"] * 100,
                "vehicle_type_id": [1] * 60 + [2] * 30 + [3] * 5 + [4] * 5,
            }
        )
        data.to_csv(tmpfile.name, index=False)
        rel_data = _rel_data_connector("trips").extract(storage_dir=tmpdir)
        rel_data.update_table_data(table="trips", data=tmpfile.name)
        yield rel_data


# These two NBA fixtures need their own temporary directories instead of
# using the tmpdir fixture because otherwise they stomp on each other
@pytest.fixture()
def source_nba():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield _setup_nba(tmpdir, synthetic=False)


@pytest.fixture()
def synthetic_nba():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield _setup_nba(tmpdir, synthetic=True)


def _setup_nba(directory: str, synthetic: bool):
    if synthetic:
        states = ["PA", "FL"]
        cities = ["Philadelphia", "Miami"]
        teams = ["Sixers", "Heat"]
    else:
        states = ["CA", "TN"]
        cities = ["Los Angeles", "Memphis"]
        teams = ["Lakers", "Grizzlies"]

    states = pd.DataFrame(data={"name": states, "id": [1, 2]})
    cities = pd.DataFrame(data={"name": cities, "id": [1, 2], "state_id": [1, 2]})
    teams = pd.DataFrame(data={"name": teams, "id": [1, 2], "city_id": [1, 2]})

    rel_data = RelationalData(directory=directory)
    rel_data.add_table(name="states", primary_key="id", data=states)
    rel_data.add_table(name="cities", primary_key="id", data=cities)
    rel_data.add_table(name="teams", primary_key="id", data=teams)
    rel_data.add_foreign_key_constraint(
        table="teams",
        constrained_columns=["city_id"],
        referred_table="cities",
        referred_columns=["id"],
    )
    rel_data.add_foreign_key_constraint(
        table="cities",
        constrained_columns=["state_id"],
        referred_table="states",
        referred_columns=["id"],
    )

    return rel_data, states, cities, teams


@pytest.fixture()
def report_json_dict() -> dict:
    return {
        "synthetic_data_quality_score": {
            "raw_score": 95.86666666666667,
            "grade": "Excellent",
            "score": 95,
        },
        "field_correlation_stability": {
            "raw_score": 0.017275336944403048,
            "grade": "Excellent",
            "score": 94,
        },
        "principal_component_stability": {
            "raw_score": 0.07294532166881013,
            "grade": "Excellent",
            "score": 100,
        },
        "field_distribution_stability": {
            "raw_score": 0.05111941886313614,
            "grade": "Excellent",
            "score": 94,
        },
        "privacy_protection_level": {
            "grade": "Good",
            "raw_score": 2,
            "score": 2,
            "outlier_filter": "Medium",
            "similarity_filter": "Disabled",
            "overfitting_protection": "Enabled",
            "differential_privacy": "Disabled",
        },
        "fatal_error": False,
        "summary": [
            {"field": "synthetic_data_quality_score", "value": 95},
            {"field": "field_correlation_stability", "value": 94},
            {"field": "principal_component_stability", "value": 100},
            {"field": "field_distribution_stability", "value": 94},
            {"field": "privacy_protection_level", "value": 2},
        ],
    }
