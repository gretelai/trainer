import os
import tempfile

import pandas as pd

from gretel_trainer.relational.relationships import RelationalData, TableProgress


def _setup_ecommerce():
    ecommerce = RelationalData()
    ecommerce.add_table("events", "id", pd.DataFrame(columns=["id", "browser", "traffic_source", "user_id"]))
    ecommerce.add_table("users", "id", pd.DataFrame(columns=["id", "first_name", "last_name"]))
    ecommerce.add_table("inventory_items", "id", pd.DataFrame(columns=["id", "sold_at", "cost", "product_id", "product_distribution_center_id"]))
    ecommerce.add_table("products", "id", pd.DataFrame(columns=["id", "name", "brand", "distribution_center_id"]))
    ecommerce.add_table("distribution_center", "id", pd.DataFrame(columns=["id", "name"]))
    ecommerce.add_table("order_items", "id", pd.DataFrame(columns=["id", "sale_price", "status", "user_id", "inventory_item_id"]))
    ecommerce.add_foreign_key("events.user_id", "users.id")
    ecommerce.add_foreign_key("order_items.user_id", "users.id")
    ecommerce.add_foreign_key("order_items.inventory_item_id", "inventory_items.id")
    ecommerce.add_foreign_key("inventory_items.product_id", "products.id")
    ecommerce.add_foreign_key("inventory_items.product_distribution_center_id", "distribution_center.id")
    ecommerce.add_foreign_key("products.distribution_center_id", "distribution_center.id")
    return ecommerce


def _setup_mutagenesis():
    mutagenesis = RelationalData()
    mutagenesis.add_table("bond", None, pd.DataFrame(columns=["type", "atom1_id", "atom2_id"]))
    mutagenesis.add_table("atom", "atom_id", pd.DataFrame(columns=["atom_id", "element", "charge", "molecule_id"]))
    mutagenesis.add_table("molecule", "molecule_id", pd.DataFrame(columns=["molecule_id", "mutagenic"]))
    mutagenesis.add_foreign_key("bond.atom1_id", "atom.atom_id")
    mutagenesis.add_foreign_key("bond.atom2_id", "atom.atom_id")
    mutagenesis.add_foreign_key("atom.molecule_id", "molecule.molecule_id")
    return mutagenesis


def test_ecommerce_relational_data():
    ecom = _setup_ecommerce()

    assert ecom.get_parents("users") == []
    assert ecom.get_parents("events") == ["users"]
    assert set(ecom.get_parents("inventory_items")) == {"products", "distribution_center"}

    users_with_ancestors = ecom.get_table_data_with_ancestors("users")
    assert set(users_with_ancestors.columns) == {
        "self|id", "self|first_name", "self|last_name"
    }
    restored_users = ecom.drop_ancestral_data(users_with_ancestors)
    assert set(restored_users.columns) == {"id", "first_name", "last_name"}

    events_with_ancestors = ecom.get_table_data_with_ancestors("events")
    assert set(events_with_ancestors.columns) == {
        "self|id", "self|browser", "self|traffic_source", "self|user_id",
        "self.user_id|id", "self.user_id|first_name", "self.user_id|last_name",
    }
    restored_events = ecom.drop_ancestral_data(events_with_ancestors)
    assert set(restored_events.columns) == {"id", "browser", "traffic_source", "user_id"}

    inventory_items_with_ancestors = ecom.get_table_data_with_ancestors("inventory_items")
    assert set(inventory_items_with_ancestors.columns) == {
        "self|id", "self|sold_at", "self|cost", "self|product_id", "self|product_distribution_center_id",
        "self.product_id|id", "self.product_id|name", "self.product_id|brand", "self.product_id|distribution_center_id",
        "self.product_distribution_center_id|id", "self.product_distribution_center_id|name",
        "self.product_id.distribution_center_id|id", "self.product_id.distribution_center_id|name",
    }
    restored_inventory_items = ecom.drop_ancestral_data(inventory_items_with_ancestors)
    assert set(restored_inventory_items.columns) == {"id", "sold_at", "cost", "product_id", "product_distribution_center_id"}

    # seed dataframes
    assert ecom.build_seed_data_for_table("users", {}) is None

    events_seed = ecom.build_seed_data_for_table("events", {"users": users_with_ancestors})
    assert events_seed is not None and set(events_seed.columns) == {
        "self.user_id|id", "self.user_id|first_name", "self.user_id|last_name",
    }

    # inventory_items_seed = ecom.build_seed_data_for_table("inventory_items", {
    #     "products": ecom.get_table_data_with_ancestors("products"),
    #     "distribution_center": ecom.get_table_data_with_ancestors("distribution_center"),
    # })
    # assert set(inventory_items_seed.columns) == {
    # # TODO
    # }


def test_mutagenesis_relational_data():
    mutagenesis = _setup_mutagenesis()

    assert mutagenesis.get_parents("bond") == ["atom"]
    assert mutagenesis.get_parents("atom") == ["molecule"]

    assert mutagenesis.get_primary_key("bond") is None
    assert mutagenesis.get_primary_key("atom") == "atom_id"

    bond_with_ancestors = mutagenesis.get_table_data_with_ancestors("bond")
    assert set(bond_with_ancestors.columns) == {
        "self|type", "self|atom1_id", "self|atom2_id",
        "self.atom1_id|atom_id", "self.atom1_id|element", "self.atom1_id|charge", "self.atom1_id|molecule_id",
        "self.atom2_id|atom_id", "self.atom2_id|element", "self.atom2_id|charge", "self.atom2_id|molecule_id",
        "self.atom1_id.molecule_id|molecule_id", "self.atom1_id.molecule_id|mutagenic",
        "self.atom2_id.molecule_id|molecule_id", "self.atom2_id.molecule_id|mutagenic",
    }
    restored_bond = mutagenesis.drop_ancestral_data(bond_with_ancestors)
    assert set(restored_bond.columns) == {"type", "atom1_id", "atom2_id"}


def test_relational_data_as_dict():
    ecom = _setup_ecommerce()
    as_dict = ecom.as_dict("test_out")

    assert as_dict["tables"] == {
        "users": {"primary_key": "id", "csv_path": "test_out/users.csv"},
        "events": {"primary_key": "id", "csv_path": "test_out/events.csv"},
        "distribution_center": {"primary_key": "id", "csv_path": "test_out/distribution_center.csv"},
        "products": {"primary_key": "id", "csv_path": "test_out/products.csv"},
        "inventory_items": {"primary_key": "id", "csv_path": "test_out/inventory_items.csv"},
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


def test_ecommerce_filesystem_serde():
    ecom = _setup_ecommerce()

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

    assert len(from_json.list_all_tables()) == 6


def test_filesystem_serde_accepts_missing_primary_keys():
    mutagenesis = _setup_mutagenesis()

    with tempfile.TemporaryDirectory() as tmp:
        mutagenesis.to_filesystem(tmp)
        from_json = RelationalData.from_filesystem(f"{tmp}/metadata.json")

    assert from_json.get_primary_key("bond") is None
    assert from_json.get_primary_key("atom") == "atom_id"


def test_walk_ecommerce():
    ecom = _setup_ecommerce()
    t = TableProgress(ecom)

    # To start, "eldest generation" tables (those with no parents / outbound foreign keys) are ready
    assert set(t.ready()) == {"users", "distribution_center"}

    t.mark_complete("users")

    # `events` was only blocked by `users`, and so now is ready
    assert set(t.ready()) == {"events", "distribution_center"}

    t.mark_complete("distribution_center")

    assert set(t.ready()) == {"events", "products"}

    t.mark_complete("events")
    t.mark_complete("products")

    assert set(t.ready()) == {"inventory_items"}

    t.mark_complete("inventory_items")

    assert set(t.ready()) == {"order_items"}

    t.mark_complete("order_items")

    assert set(t.ready()) == set()
