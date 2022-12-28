import os
import tempfile

import pandas as pd
import pandas.testing as pdtest

from gretel_trainer.relational.core import RelationalData


def test_ecommerce_relational_data(ecom):
    assert ecom.get_parents("users") == []
    assert ecom.get_parents("events") == ["users"]
    assert set(ecom.get_parents("inventory_items")) == {
        "products",
        "distribution_center",
    }

    users_with_ancestors = ecom.get_table_data_with_ancestors("users")
    assert set(users_with_ancestors.columns) == {
        "self|id",
        "self|first_name",
        "self|last_name",
    }
    restored_users = ecom.drop_ancestral_data(users_with_ancestors)
    assert set(restored_users.columns) == {"id", "first_name", "last_name"}

    events_with_ancestors = ecom.get_table_data_with_ancestors("events")
    assert set(events_with_ancestors.columns) == {
        "self|id",
        "self|browser",
        "self|traffic_source",
        "self|user_id",
        "self.user_id|id",
        "self.user_id|first_name",
        "self.user_id|last_name",
    }
    restored_events = ecom.drop_ancestral_data(events_with_ancestors)
    assert set(restored_events.columns) == {
        "id",
        "browser",
        "traffic_source",
        "user_id",
    }

    inventory_items_with_ancestors = ecom.get_table_data_with_ancestors(
        "inventory_items"
    )
    assert set(inventory_items_with_ancestors.columns) == {
        "self|id",
        "self|sold_at",
        "self|cost",
        "self|product_id",
        "self|product_distribution_center_id",
        "self.product_id|id",
        "self.product_id|name",
        "self.product_id|brand",
        "self.product_id|distribution_center_id",
        "self.product_distribution_center_id|id",
        "self.product_distribution_center_id|name",
        "self.product_id.distribution_center_id|id",
        "self.product_id.distribution_center_id|name",
    }
    restored_inventory_items = ecom.drop_ancestral_data(inventory_items_with_ancestors)
    assert set(restored_inventory_items.columns) == {
        "id",
        "sold_at",
        "cost",
        "product_id",
        "product_distribution_center_id",
    }


def test_mutagenesis_relational_data(mutagenesis):
    assert mutagenesis.get_parents("bond") == ["atom"]
    assert mutagenesis.get_parents("atom") == ["molecule"]

    assert mutagenesis.get_primary_key("bond") is None
    assert mutagenesis.get_primary_key("atom") == "atom_id"

    bond_with_ancestors = mutagenesis.get_table_data_with_ancestors("bond")
    assert set(bond_with_ancestors.columns) == {
        "self|type",
        "self|atom1_id",
        "self|atom2_id",
        "self.atom1_id|atom_id",
        "self.atom1_id|element",
        "self.atom1_id|charge",
        "self.atom1_id|molecule_id",
        "self.atom2_id|atom_id",
        "self.atom2_id|element",
        "self.atom2_id|charge",
        "self.atom2_id|molecule_id",
        "self.atom1_id.molecule_id|molecule_id",
        "self.atom1_id.molecule_id|mutagenic",
        "self.atom2_id.molecule_id|molecule_id",
        "self.atom2_id.molecule_id|mutagenic",
    }
    restored_bond = mutagenesis.drop_ancestral_data(bond_with_ancestors)
    assert set(restored_bond.columns) == {"type", "atom1_id", "atom2_id"}


def test_ancestral_data_from_different_tablesets(source_nba, synthetic_nba):
    rel_data, _, _, _ = source_nba
    _, custom_states, custom_cities, custom_teams = synthetic_nba

    # By default, get data from source
    source_teams_with_ancestors = rel_data.get_table_data_with_ancestors("teams")
    assert set(source_teams_with_ancestors["self|name"]) == {"Lakers", "Grizzlies"}

    custom_tableset = {
        "states": custom_states,
        "cities": custom_cities,
        "teams": custom_teams,
    }

    # Optionally provide a different tableset
    custom_teams_with_ancestors = rel_data.get_table_data_with_ancestors(
        "teams", custom_tableset
    )
    assert set(custom_teams_with_ancestors["self|name"]) == {"Sixers", "Heat"}


def test_list_ancestral_keys(mutagenesis):
    assert set(mutagenesis.list_multigenerational_keys("bond")) == {
        "self|atom1_id",
        "self|atom2_id",
        "self.atom1_id|atom_id",
        "self.atom1_id|molecule_id",
        "self.atom2_id|atom_id",
        "self.atom2_id|molecule_id",
        "self.atom1_id.molecule_id|molecule_id",
        "self.atom2_id.molecule_id|molecule_id",
    }


def test_whether_column_is_ancestral(mutagenesis):
    assert mutagenesis.is_ancestral_column("self|atom1_id") is False
    assert mutagenesis.is_ancestral_column("self.atom1_id|atom1_id")
    assert mutagenesis.is_ancestral_column("self.atom1_id.molecule_id|atom1_id")


def test_relational_data_as_dict(ecom):
    as_dict = ecom.as_dict("test_out")

    assert as_dict["tables"] == {
        "users": {"primary_key": "id", "csv_path": "test_out/users.csv"},
        "events": {"primary_key": "id", "csv_path": "test_out/events.csv"},
        "distribution_center": {
            "primary_key": "id",
            "csv_path": "test_out/distribution_center.csv",
        },
        "products": {"primary_key": "id", "csv_path": "test_out/products.csv"},
        "inventory_items": {
            "primary_key": "id",
            "csv_path": "test_out/inventory_items.csv",
        },
        "order_items": {"primary_key": "id", "csv_path": "test_out/order_items.csv"},
    }
    assert set(as_dict["foreign_keys"]) == {
        ("events.user_id", "users.id"),
        ("order_items.user_id", "users.id"),
        ("order_items.inventory_item_id", "inventory_items.id"),
        ("inventory_items.product_id", "products.id"),
        ("inventory_items.product_distribution_center_id", "distribution_center.id"),
        ("products.distribution_center_id", "distribution_center.id"),
    }


def test_ecommerce_filesystem_serde(ecom):
    with tempfile.TemporaryDirectory() as tmp:
        ecom.to_filesystem(tmp)

        expected_files = [
            f"{tmp}/metadata.json",
            f"{tmp}/events.csv",
            f"{tmp}/users.csv",
            f"{tmp}/distribution_center.csv",
            f"{tmp}/products.csv",
            f"{tmp}/inventory_items.csv",
            f"{tmp}/order_items.csv",
        ]
        for expected_file in expected_files:
            assert os.path.exists(expected_file)

        from_json = RelationalData.from_filesystem(f"{tmp}/metadata.json")

    for table in ecom.list_all_tables():
        pdtest.assert_frame_equal(
            ecom.get_table_data(table), from_json.get_table_data(table)
        )
        assert ecom.get_parents(table) == from_json.get_parents(table)
        assert ecom.get_foreign_keys(table) == from_json.get_foreign_keys(table)


def test_filesystem_serde_accepts_missing_primary_keys(mutagenesis):
    with tempfile.TemporaryDirectory() as tmp:
        mutagenesis.to_filesystem(tmp)
        from_json = RelationalData.from_filesystem(f"{tmp}/metadata.json")

    assert from_json.get_primary_key("bond") is None
    assert from_json.get_primary_key("atom") == "atom_id"


def test_debug_summary(ecom, mutagenesis):
    assert ecom.debug_summary() == {
        "foreign_key_count": 6,
        "max_depth": 3,
        "table_count": 6,
        "tables": {
            "users": {"foreign_key_count": 0, "primary_key": "id"},
            "events": {"foreign_key_count": 1, "primary_key": "id"},
            "distribution_center": {"foreign_key_count": 0, "primary_key": "id"},
            "products": {"foreign_key_count": 1, "primary_key": "id"},
            "inventory_items": {"foreign_key_count": 2, "primary_key": "id"},
            "order_items": {"foreign_key_count": 2, "primary_key": "id"},
        },
    }

    assert mutagenesis.debug_summary() == {
        "foreign_key_count": 3,
        "max_depth": 2,
        "table_count": 3,
        "tables": {
            "bond": {"foreign_key_count": 2, "primary_key": None},
            "atom": {"foreign_key_count": 1, "primary_key": "atom_id"},
            "molecule": {"foreign_key_count": 0, "primary_key": "molecule_id"},
        },
    }


def test_ancestral_foreign_key_maps(ecom):
    events_afk_maps = ecom.get_ancestral_foreign_key_maps("events")
    assert events_afk_maps == [("self|user_id", "self.user_id|id")]

    inventory_items_afk_maps = ecom.get_ancestral_foreign_key_maps("inventory_items")
    assert set(inventory_items_afk_maps) == {
        ("self|product_id", "self.product_id|id"),
        (
            "self|product_distribution_center_id",
            "self.product_distribution_center_id|id",
        ),
    }


def test_building_seeds(source_nba, synthetic_nba):
    source_nba = source_nba[0]
    synthetic_nba = synthetic_nba[0]

    # Cannot build seed data for tables with no parents
    assert source_nba.build_seed_data_for_table("states") is None
    assert source_nba.build_seed_data_for_table("states", {}) is None

    # When no specific ancestor data is provided, uses source data
    nba_source_city_seed = source_nba.build_seed_data_for_table("cities")
    pdtest.assert_frame_equal(
        nba_source_city_seed,
        pd.DataFrame(
            data={
                "self.state_id|name": ["CA", "TN"],
                "self.state_id|id": [1, 2],
            }
        ),
    )

    # When ancestor data is provided, uses that tableset
    synth_tables = {"states": synthetic_nba.get_table_data_with_ancestors("states")}
    nba_synth_city_seed = source_nba.build_seed_data_for_table("cities", synth_tables)
    pdtest.assert_frame_equal(
        nba_synth_city_seed,
        pd.DataFrame(
            data={
                "self.state_id|name": ["PA", "FL"],
                "self.state_id|id": [1, 2],
            }
        ),
    )

    # Seeds go back multiple generations
    teams_seed = source_nba.build_seed_data_for_table("teams")
    assert set(teams_seed.columns) == {
        "self.city_id|id",
        "self.city_id|name",
        "self.city_id|state_id",
        "self.city_id.state_id|id",
        "self.city_id.state_id|name",
    }
