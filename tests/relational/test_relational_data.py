import pandas as pd
import pytest

from gretel_trainer.relational.core import MultiTableException


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


def test_column_metadata(pets, tmpfile):
    assert pets.get_table_columns("humans") == ["id", "name", "city"]

    # Name is a highly unique categorical field, so is excluded
    assert pets.get_safe_ancestral_seed_columns("humans") == {"id", "city"}

    # Update the table data such that:
    # - id is highly unique categorical (_ to force string instead of int), but still the PK
    # - name is no longer highly unique
    # - city is highly NaN
    pd.DataFrame(
        data={
            "id": ["1_", "2_", "3_"],
            "name": ["n", "n", "n"],
            "city": [None, None, "Chicago"],
        }
    ).to_csv(tmpfile.name, index=False)
    pets.update_table_data(
        "humans",
        tmpfile.name,
    )

    assert pets.get_safe_ancestral_seed_columns("humans") == {"id", "name"}

    # Resetting the primary key refreshes the cache state
    # In this case, since id is no longer the PK and is highly unique, it is excluded
    pets.set_primary_key(table="humans", primary_key=None)
    assert pets.get_safe_ancestral_seed_columns("humans") == {"name"}

    # Reset back to normal
    pets.set_primary_key(table="humans", primary_key="id")

    # Setting a column as a foreign key ensures it is included
    pets.add_foreign_key_constraint(
        table="humans",
        constrained_columns=["city"],
        referred_table="pets",
        referred_columns=["id"],
    )
    assert pets.get_safe_ancestral_seed_columns("humans") == {"id", "name", "city"}

    # Removing a foreign key refreshes the cache state
    pets.remove_foreign_key_constraint("humans", ["city"])
    assert pets.get_safe_ancestral_seed_columns("humans") == {"id", "name"}


def test_adding_and_removing_foreign_keys(pets):
    # pets has a foreign key defined out of the box.
    # First lets successfully remove it and re-add it.
    assert len(pets.get_foreign_keys("pets")) == 1

    pets.remove_foreign_key_constraint(table="pets", constrained_columns=["human_id"])
    assert len(pets.get_foreign_keys("pets")) == 0

    pets.add_foreign_key_constraint(
        table="pets",
        constrained_columns=["human_id"],
        referred_table="humans",
        referred_columns=["id"],
    )
    assert len(pets.get_foreign_keys("pets")) == 1

    # Now we'll make some assertions about our defense

    # Cannot add to an unrecognized table
    with pytest.raises(MultiTableException):
        pets.add_foreign_key_constraint(
            table="unrecognized",
            constrained_columns=["user_id"],
            referred_table="humans",
            referred_columns=["id"],
        )

    # Cannot add to an unrecognized referred table
    with pytest.raises(MultiTableException):
        pets.add_foreign_key_constraint(
            table="pets",
            constrained_columns=["human_id"],
            referred_table="unrecognized",
            referred_columns=["id"],
        )

    # Cannot add unrecognized columns
    with pytest.raises(MultiTableException):
        pets.add_foreign_key_constraint(
            table="pets",
            constrained_columns=["human_id"],
            referred_table="humans",
            referred_columns=["unrecognized"],
        )
    with pytest.raises(MultiTableException):
        pets.add_foreign_key_constraint(
            table="pets",
            constrained_columns=["unrecognized"],
            referred_table="humans",
            referred_columns=["id"],
        )

    # Cannot remove from unrecognized table
    with pytest.raises(MultiTableException):
        pets.remove_foreign_key_constraint(
            table="unrecognized", constrained_columns=["id"]
        )

    # Cannot remove a non-existent key
    with pytest.raises(MultiTableException):
        pets.remove_foreign_key_constraint(table="pets", constrained_columns=["id"])


def test_add_remove_foreign_key_shorthand(pets):
    assert len(pets.get_foreign_keys("pets")) == 1

    pets.remove_foreign_key("pets.human_id")
    assert len(pets.get_foreign_keys("pets")) == 0

    pets.add_foreign_key(foreign_key="pets.human_id", referencing="humans.id")
    assert len(pets.get_foreign_keys("pets")) == 1


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


def test_get_subset_of_data(pets):
    normal_length = len(pets.get_table_data("humans"))
    subset = pets.get_table_data("humans", ["name", "city"])
    assert list(subset.columns) == ["name", "city"]
    assert len(subset) == normal_length


def test_list_tables_parents_before_children(ecom):
    def in_order(col, t1, t2):
        return col.index(t1) < col.index(t2)

    tables = ecom.list_tables_parents_before_children()
    assert in_order(tables, "users", "events")
    assert in_order(tables, "distribution_center", "products")
    assert in_order(tables, "distribution_center", "inventory_items")
    assert in_order(tables, "products", "inventory_items")
    assert in_order(tables, "inventory_items", "order_items")
    assert in_order(tables, "users", "order_items")


def test_debug_summary(ecom, mutagenesis):
    assert ecom.debug_summary() == {
        "foreign_key_count": 6,
        "max_depth": 3,
        "public_table_count": 6,
        "invented_table_count": 0,
        "tables": {
            "users": {
                "column_count": 3,
                "primary_key": ["id"],
                "foreign_key_count": 0,
                "foreign_keys": [],
                "is_invented_table": False,
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
                "is_invented_table": False,
            },
            "distribution_center": {
                "column_count": 2,
                "primary_key": ["id"],
                "foreign_key_count": 0,
                "foreign_keys": [],
                "is_invented_table": False,
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
                "is_invented_table": False,
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
                "is_invented_table": False,
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
                "is_invented_table": False,
            },
        },
    }

    assert mutagenesis.debug_summary() == {
        "foreign_key_count": 3,
        "max_depth": 2,
        "public_table_count": 3,
        "invented_table_count": 0,
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
                "is_invented_table": False,
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
                "is_invented_table": False,
            },
            "molecule": {
                "column_count": 2,
                "primary_key": ["molecule_id"],
                "foreign_key_count": 0,
                "foreign_keys": [],
                "is_invented_table": False,
            },
        },
    }
