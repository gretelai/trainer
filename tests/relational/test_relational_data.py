import os
import tempfile

import pandas as pd
import pytest

from gretel_trainer.relational.core import MultiTableException, RelationalData


def test_ecommerce_relational_data(ecom):
    assert ecom.get_parents("users") == []
    assert ecom.get_parents("events") == ["users"]
    assert set(ecom.get_parents("inventory_items")) == {
        "products",
        "distribution_center",
    }

    # get_parents goes back one generation,
    # get_ancestors goes back all generations
    assert set(ecom.get_parents("order_items")) == {
        "users",
        "inventory_items",
    }
    assert set(ecom.get_ancestors("order_items")) == {
        "users",
        "inventory_items",
        "products",
        "distribution_center",
    }

    # get_descendants goes forward all generations
    assert set(ecom.get_descendants("products")) == {
        "inventory_items",
        "order_items",
    }


def test_mutagenesis_relational_data(mutagenesis):
    assert mutagenesis.get_parents("bond") == ["atom"]
    assert mutagenesis.get_parents("atom") == ["molecule"]

    assert mutagenesis.get_primary_key("bond") == ["atom1_id", "atom2_id"]
    assert mutagenesis.get_primary_key("atom") == ["atom_id"]

    assert set(mutagenesis.get_all_key_columns("bond")) == {"atom1_id", "atom2_id"}
    assert set(mutagenesis.get_all_key_columns("atom")) == {"atom_id", "molecule_id"}


def test_column_metadata(pets):
    assert pets.get_table_columns("humans") == {"id", "name", "city"}

    # Name is a highly unique categorical field, so is excluded
    assert pets.get_safe_ancestral_seed_columns("humans") == {"id", "city"}

    # Update the table data such that:
    # - id is highly unique categorical, but still the PK
    # - name is no longer highly unique
    # - city is highly NaN
    pets.update_table_data(
        "humans",
        pd.DataFrame(
            data={
                "id": ["1", "2", "3"],
                "name": ["n", "n", "n"],
                "city": [None, None, "Chicago"],
            }
        ),
    )
    assert pets.get_safe_ancestral_seed_columns("humans") == {"id", "name"}

    # Resetting the primary key refreshes the cache state
    # In this case, since id is no longer the PK and is highly unique, it is excluded
    pets.set_primary_key(table="humans", primary_key=None)
    assert pets.get_safe_ancestral_seed_columns("humans") == {"name"}

    # Reset back to normal
    pets.set_primary_key(table="humans", primary_key="id")

    # Setting a column as a foreign key ensures it is included
    pets.add_foreign_key(
        table="humans",
        constrained_columns=["city"],
        referred_table="pets",
        referred_columns=["id"],
    )
    assert pets.get_safe_ancestral_seed_columns("humans") == {"id", "name", "city"}

    # Setting a column as a foreign key ensures it is included
    pets.remove_foreign_key("humans", ["city"])
    assert pets.get_safe_ancestral_seed_columns("humans") == {"id", "name"}


def test_adding_and_removing_foreign_keys():
    rel_data = RelationalData()
    rel_data.add_table(
        name="users", primary_key="id", data=pd.DataFrame(data={"id": [1, 2, 3]})
    )
    rel_data.add_table(
        name="events",
        primary_key="id",
        data=pd.DataFrame(data={"id": [1, 2, 3], "user_id": [1, 2, 3]}),
    )

    # Cannot add to an unrecognized table
    with pytest.raises(MultiTableException):
        rel_data.add_foreign_key(
            table="unrecognized",
            constrained_columns=["user_id"],
            referred_table="users",
            referred_columns=["id"],
        )

    # Cannot add to an unrecognized referred table
    with pytest.raises(MultiTableException):
        rel_data.add_foreign_key(
            table="events",
            constrained_columns=["user_id"],
            referred_table="unrecognized",
            referred_columns=["id"],
        )

    # Cannot add unrecognized columns
    with pytest.raises(MultiTableException):
        rel_data.add_foreign_key(
            table="events",
            constrained_columns=["user_id"],
            referred_table="users",
            referred_columns=["unrecognized"],
        )
    with pytest.raises(MultiTableException):
        rel_data.add_foreign_key(
            table="events",
            constrained_columns=["unrecognized"],
            referred_table="users",
            referred_columns=["id"],
        )

    # Successful add
    rel_data.add_foreign_key(
        table="events",
        constrained_columns=["user_id"],
        referred_table="users",
        referred_columns=["id"],
    )
    assert len(rel_data.get_foreign_keys("events")) == 1

    # Cannot remove from unrecognized table
    with pytest.raises(MultiTableException):
        rel_data.remove_foreign_key(table="unrecognized", constrained_columns=["id"])

    # Cannot remove a non-existent key
    with pytest.raises(MultiTableException):
        rel_data.remove_foreign_key(table="events", constrained_columns=["id"])

    # Successful remove
    rel_data.remove_foreign_key(table="events", constrained_columns=["user_id"])
    assert len(rel_data.get_foreign_keys("events")) == 0


def test_set_primary_key(ecom):
    assert ecom.get_primary_key("users") == ["id"]

    ecom.set_primary_key(table="users", primary_key=None)
    assert ecom.get_primary_key("users") == []

    ecom.set_primary_key(table="users", primary_key=["first_name", "last_name"])
    assert ecom.get_primary_key("users") == ["first_name", "last_name"]

    ecom.set_primary_key(table="users", primary_key="id")
    assert ecom.get_primary_key("users") == ["id"]

    # Can't set primary key on an unknown table
    with pytest.raises(MultiTableException):
        ecom.set_primary_key(table="not_a_table", primary_key="id")

    # Can't set primary key to a non-existent column
    with pytest.raises(MultiTableException):
        ecom.set_primary_key(table="users", primary_key="not_a_column")


def test_relational_data_as_dict(ecom):
    as_dict = ecom.as_dict("test_out")

    assert as_dict["tables"] == {
        "users": {"primary_key": ["id"], "csv_path": "test_out/users.csv"},
        "events": {"primary_key": ["id"], "csv_path": "test_out/events.csv"},
        "distribution_center": {
            "primary_key": ["id"],
            "csv_path": "test_out/distribution_center.csv",
        },
        "products": {"primary_key": ["id"], "csv_path": "test_out/products.csv"},
        "inventory_items": {
            "primary_key": ["id"],
            "csv_path": "test_out/inventory_items.csv",
        },
        "order_items": {"primary_key": ["id"], "csv_path": "test_out/order_items.csv"},
    }
    expected_foreign_keys = [
        {
            "table": "events",
            "constrained_columns": ["user_id"],
            "referred_table": "users",
            "referred_columns": ["id"],
        },
        {
            "table": "order_items",
            "constrained_columns": ["user_id"],
            "referred_table": "users",
            "referred_columns": ["id"],
        },
        {
            "table": "order_items",
            "constrained_columns": ["inventory_item_id"],
            "referred_table": "inventory_items",
            "referred_columns": ["id"],
        },
        {
            "table": "inventory_items",
            "constrained_columns": ["product_id"],
            "referred_table": "products",
            "referred_columns": ["id"],
        },
        {
            "table": "inventory_items",
            "constrained_columns": ["product_distribution_center_id"],
            "referred_table": "distribution_center",
            "referred_columns": ["id"],
        },
        {
            "table": "products",
            "constrained_columns": ["distribution_center_id"],
            "referred_table": "distribution_center",
            "referred_columns": ["id"],
        },
    ]
    for expected_fk in expected_foreign_keys:
        assert expected_fk in as_dict["foreign_keys"]


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
        assert set(ecom.get_table_data(table).columns) == set(
            from_json.get_table_data(table).columns
        )
        assert ecom.get_parents(table) == from_json.get_parents(table)
        assert ecom.get_foreign_keys(table) == from_json.get_foreign_keys(table)


def test_filesystem_serde_accepts_composite_primary_keys(mutagenesis):
    with tempfile.TemporaryDirectory() as tmp:
        mutagenesis.to_filesystem(tmp)
        from_json = RelationalData.from_filesystem(f"{tmp}/metadata.json")

    assert from_json.get_primary_key("bond") == ["atom1_id", "atom2_id"]
    assert from_json.get_primary_key("atom") == ["atom_id"]


def test_debug_summary(ecom, mutagenesis):
    assert ecom.debug_summary() == {
        "foreign_key_count": 6,
        "max_depth": 3,
        "table_count": 6,
        "tables": {
            "users": {
                "column_count": 3,
                "primary_key": ["id"],
                "foreign_key_count": 0,
                "foreign_keys": [],
            },
            "events": {
                "column_count": 4,
                "primary_key": ["id"],
                "foreign_key_count": 1,
                "foreign_keys": [
                    {
                        "columns": ["user_id"],
                        "parent_table_name": "users",
                        "parent_columns": ["id"],
                    }
                ],
            },
            "distribution_center": {
                "column_count": 2,
                "primary_key": ["id"],
                "foreign_key_count": 0,
                "foreign_keys": [],
            },
            "products": {
                "column_count": 4,
                "primary_key": ["id"],
                "foreign_key_count": 1,
                "foreign_keys": [
                    {
                        "columns": ["distribution_center_id"],
                        "parent_table_name": "distribution_center",
                        "parent_columns": ["id"],
                    }
                ],
            },
            "inventory_items": {
                "column_count": 5,
                "primary_key": ["id"],
                "foreign_key_count": 2,
                "foreign_keys": [
                    {
                        "columns": ["product_id"],
                        "parent_table_name": "products",
                        "parent_columns": ["id"],
                    },
                    {
                        "columns": ["product_distribution_center_id"],
                        "parent_table_name": "distribution_center",
                        "parent_columns": ["id"],
                    },
                ],
            },
            "order_items": {
                "column_count": 5,
                "primary_key": ["id"],
                "foreign_key_count": 2,
                "foreign_keys": [
                    {
                        "columns": ["user_id"],
                        "parent_table_name": "users",
                        "parent_columns": ["id"],
                    },
                    {
                        "columns": ["inventory_item_id"],
                        "parent_table_name": "inventory_items",
                        "parent_columns": ["id"],
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
                "column_count": 3,
                "primary_key": ["atom1_id", "atom2_id"],
                "foreign_key_count": 2,
                "foreign_keys": [
                    {
                        "columns": ["atom1_id"],
                        "parent_table_name": "atom",
                        "parent_columns": ["atom_id"],
                    },
                    {
                        "columns": ["atom2_id"],
                        "parent_table_name": "atom",
                        "parent_columns": ["atom_id"],
                    },
                ],
            },
            "atom": {
                "column_count": 4,
                "primary_key": ["atom_id"],
                "foreign_key_count": 1,
                "foreign_keys": [
                    {
                        "columns": ["molecule_id"],
                        "parent_table_name": "molecule",
                        "parent_columns": ["molecule_id"],
                    }
                ],
            },
            "molecule": {
                "column_count": 2,
                "primary_key": ["molecule_id"],
                "foreign_key_count": 0,
                "foreign_keys": [],
            },
        },
    }
