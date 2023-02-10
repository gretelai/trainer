import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from sqlalchemy import create_engine

from gretel_trainer.relational.connectors import Connector
from gretel_trainer.relational.core import RelationalData

EXAMPLE_DBS = Path(__file__).parent.resolve() / "example_dbs"


def rel_data_from_example_db(name):
    con = sqlite3.connect(f"file:{name}?mode=memory&cache=shared")
    cur = con.cursor()
    with open(EXAMPLE_DBS / f"{name}.sql") as f:
        cur.executescript(f.read())
    connector = Connector(
        create_engine(f"sqlite:///file:{name}?mode=memory&cache=shared&uri=true")
    )
    return connector.extract()


@pytest.fixture()
def example_dbs():
    return EXAMPLE_DBS


@pytest.fixture()
def pets():
    return rel_data_from_example_db("pets")


@pytest.fixture()
def ecom():
    return rel_data_from_example_db("ecom")


@pytest.fixture()
def mutagenesis():
    return rel_data_from_example_db("mutagenesis")


@pytest.fixture()
def trips():
    rel_data = rel_data_from_example_db("trips")
    rel_data.update_table_data(
        table="trips",
        data=pd.DataFrame(
            data={
                "id": list(range(100)),
                "vehicle_type_id": [1] * 60 + [2] * 30 + [3] * 5 + [4] * 5,
                "purpose": ["work"] * 100,
            }
        ),
    )
    return rel_data


@pytest.fixture()
def source_nba():
    return _setup_nba(synthetic=False)


@pytest.fixture()
def synthetic_nba():
    return _setup_nba(synthetic=True)


def _setup_nba(synthetic: bool):
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

    rel_data = RelationalData()
    rel_data.add_table(name="states", primary_key="id", data=states)
    rel_data.add_table(name="cities", primary_key="id", data=cities)
    rel_data.add_table(name="teams", primary_key="id", data=teams)
    rel_data.add_foreign_key(foreign_key="teams.city_id", referencing="cities.id")
    rel_data.add_foreign_key(foreign_key="cities.state_id", referencing="states.id")

    return rel_data, states, cities, teams
