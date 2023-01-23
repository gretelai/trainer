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


def test_mutagenesis_relational_data(mutagenesis):
    assert mutagenesis.get_parents("bond") == ["atom"]
    assert mutagenesis.get_parents("atom") == ["molecule"]

    assert mutagenesis.get_primary_key("bond") is None
    assert mutagenesis.get_primary_key("atom") == "atom_id"


def test_add_foreign_key_checks_if_tables_exist():
    rel_data = RelationalData()
    rel_data.add_table("users", "id", pd.DataFrame())
    rel_data.add_table("events", "id", pd.DataFrame())

    # attempt to add a foreign key to an unrecognized table
    rel_data.add_foreign_key("events.user_id", "USR.id")
    assert len(rel_data.get_foreign_keys("events")) == 0
    assert set(rel_data.list_all_tables()) == {"users", "events"}

    # again from the opposite side
    rel_data.add_foreign_key("EVNT.user_id", "users.id")
    assert set(rel_data.list_all_tables()) == {"users", "events"}

    # add a foreign key correctly
    rel_data.add_foreign_key("events.user_id", "users.id")
    assert len(rel_data.get_foreign_keys("events")) == 1


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
            "users": {
                "primary_key": "id",
                "foreign_key_count": 0,
                "foreign_keys": [],
            },
            "events": {
                "primary_key": "id",
                "foreign_key_count": 1,
                "foreign_keys": [
                    {
                        "column_name": "user_id",
                        "parent_column_name": "id",
                        "parent_table_name": "users",
                    }
                ],
            },
            "distribution_center": {
                "primary_key": "id",
                "foreign_key_count": 0,
                "foreign_keys": [],
            },
            "products": {
                "primary_key": "id",
                "foreign_key_count": 1,
                "foreign_keys": [
                    {
                        "column_name": "distribution_center_id",
                        "parent_column_name": "id",
                        "parent_table_name": "distribution_center",
                    }
                ],
            },
            "inventory_items": {
                "primary_key": "id",
                "foreign_key_count": 2,
                "foreign_keys": [
                    {
                        "column_name": "product_id",
                        "parent_column_name": "id",
                        "parent_table_name": "products",
                    },
                    {
                        "column_name": "product_distribution_center_id",
                        "parent_column_name": "id",
                        "parent_table_name": "distribution_center",
                    },
                ],
            },
            "order_items": {
                "primary_key": "id",
                "foreign_key_count": 2,
                "foreign_keys": [
                    {
                        "column_name": "user_id",
                        "parent_column_name": "id",
                        "parent_table_name": "users",
                    },
                    {
                        "column_name": "inventory_item_id",
                        "parent_column_name": "id",
                        "parent_table_name": "inventory_items",
                    },
                ],
            },
        },
    }

    assert mutagenesis.debug_summary() == {
        "foreign_key_count": 3,
        "max_depth": 2,
        "table_count": 3,
        "tables": {
            "bond": {
                "primary_key": None,
                "foreign_key_count": 2,
                "foreign_keys": [
                    {
                        "column_name": "atom1_id",
                        "parent_column_name": "atom_id",
                        "parent_table_name": "atom",
                    },
                    {
                        "column_name": "atom2_id",
                        "parent_column_name": "atom_id",
                        "parent_table_name": "atom",
                    },
                ],
            },
            "atom": {
                "primary_key": "atom_id",
                "foreign_key_count": 1,
                "foreign_keys": [
                    {
                        "column_name": "molecule_id",
                        "parent_column_name": "molecule_id",
                        "parent_table_name": "molecule",
                    }
                ],
            },
            "molecule": {
                "primary_key": "molecule_id",
                "foreign_key_count": 0,
                "foreign_keys": [],
            },
        },
    }
