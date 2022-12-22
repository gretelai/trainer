import pandas as pd
import pytest
from unittest.mock import Mock, patch

from gretel_trainer.relational.core import RelationalData


@pytest.fixture()
def configured_session():
    # Need to patch configure_session in two spots because MultiTable calls it first
    # (before any work is done) and then Trainer instances call it internally
    with patch("gretel_trainer.relational.multi_table.configure_session"), patch(
        "gretel_trainer.trainer.configure_session"
    ):
        yield


@pytest.fixture()
def pets():
    humans = pd.DataFrame(
        data={
            "name": ["John", "Paul", "George", "Ringo", "Billy"],
            "city": ["Liverpool", "Liverpool", "Liverpool", "Liverpool", "Houston"],
            "id": [1, 2, 3, 4, 5],
        }
    )
    pets = pd.DataFrame(
        data={
            "name": ["Lennon", "McCartney", "Harrison", "Starr", "Preston"],
            "age": [6, 14, 8, 7, 2],
            "id": [1, 2, 3, 4, 5],
            "human_id": [1, 2, 3, 4, 5],
        }
    )
    rel_data = RelationalData()
    rel_data.add_table("humans", "id", humans)
    rel_data.add_table("pets", "id", pets)
    rel_data.add_foreign_key("pets.human_id", "humans.id")
    return rel_data


@pytest.fixture()
def ecom():
    ecommerce = RelationalData()
    ecommerce.add_table(
        "events",
        "id",
        pd.DataFrame(columns=["id", "browser", "traffic_source", "user_id"]),
    )
    ecommerce.add_table(
        "users", "id", pd.DataFrame(columns=["id", "first_name", "last_name"])
    )
    ecommerce.add_table(
        "inventory_items",
        "id",
        pd.DataFrame(
            columns=[
                "id",
                "sold_at",
                "cost",
                "product_id",
                "product_distribution_center_id",
            ]
        ),
    )
    ecommerce.add_table(
        "products",
        "id",
        pd.DataFrame(columns=["id", "name", "brand", "distribution_center_id"]),
    )
    ecommerce.add_table(
        "distribution_center", "id", pd.DataFrame(columns=["id", "name"])
    )
    ecommerce.add_table(
        "order_items",
        "id",
        pd.DataFrame(
            columns=["id", "sale_price", "status", "user_id", "inventory_item_id"]
        ),
    )
    ecommerce.add_foreign_key("events.user_id", "users.id")
    ecommerce.add_foreign_key("order_items.user_id", "users.id")
    ecommerce.add_foreign_key("order_items.inventory_item_id", "inventory_items.id")
    ecommerce.add_foreign_key("inventory_items.product_id", "products.id")
    ecommerce.add_foreign_key(
        "inventory_items.product_distribution_center_id", "distribution_center.id"
    )
    ecommerce.add_foreign_key(
        "products.distribution_center_id", "distribution_center.id"
    )
    return ecommerce


@pytest.fixture()
def mutagenesis():
    mutagenesis = RelationalData()
    mutagenesis.add_table(
        "bond", None, pd.DataFrame(columns=["type", "atom1_id", "atom2_id"])
    )
    mutagenesis.add_table(
        "atom",
        "atom_id",
        pd.DataFrame(columns=["atom_id", "element", "charge", "molecule_id"]),
    )
    mutagenesis.add_table(
        "molecule", "molecule_id", pd.DataFrame(columns=["molecule_id", "mutagenic"])
    )
    mutagenesis.add_foreign_key("bond.atom1_id", "atom.atom_id")
    mutagenesis.add_foreign_key("bond.atom2_id", "atom.atom_id")
    mutagenesis.add_foreign_key("atom.molecule_id", "molecule.molecule_id")
    return mutagenesis


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
    rel_data.add_table("states", "id", states)
    rel_data.add_table("cities", "id", cities)
    rel_data.add_table("teams", "id", teams)
    rel_data.add_foreign_key("teams.city_id", "cities.id")
    rel_data.add_foreign_key("cities.state_id", "states.id")

    return rel_data, states, cities, teams
