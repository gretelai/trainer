import os
import tempfile
from unittest.mock import patch

import pandas as pd
import pandas.testing as pdtest
import pytest

from gretel_trainer.relational.core import (
    ForeignKey,
    MultiTableException,
    RelationalData,
    Scope,
)


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
        rel_data.add_foreign_key_constraint(
            table="unrecognized",
            constrained_columns=["user_id"],
            referred_table="users",
            referred_columns=["id"],
        )

    # Cannot add to an unrecognized referred table
    with pytest.raises(MultiTableException):
        rel_data.add_foreign_key_constraint(
            table="events",
            constrained_columns=["user_id"],
            referred_table="unrecognized",
            referred_columns=["id"],
        )

    # Cannot add unrecognized columns
    with pytest.raises(MultiTableException):
        rel_data.add_foreign_key_constraint(
            table="events",
            constrained_columns=["user_id"],
            referred_table="users",
            referred_columns=["unrecognized"],
        )
    with pytest.raises(MultiTableException):
        rel_data.add_foreign_key_constraint(
            table="events",
            constrained_columns=["unrecognized"],
            referred_table="users",
            referred_columns=["id"],
        )

    # Successful add
    rel_data.add_foreign_key_constraint(
        table="events",
        constrained_columns=["user_id"],
        referred_table="users",
        referred_columns=["id"],
    )
    assert len(rel_data.get_foreign_keys("events")) == 1

    # Cannot remove from unrecognized table
    with pytest.raises(MultiTableException):
        rel_data.remove_foreign_key_constraint(
            table="unrecognized", constrained_columns=["id"]
        )

    # Cannot remove a non-existent key
    with pytest.raises(MultiTableException):
        rel_data.remove_foreign_key_constraint(
            table="events", constrained_columns=["id"]
        )

    # Successful remove
    rel_data.remove_foreign_key_constraint(
        table="events", constrained_columns=["user_id"]
    )
    assert len(rel_data.get_foreign_keys("events")) == 0


def test_add_remove_foreign_key_shorthand():
    rel_data = RelationalData()
    rel_data.add_table(
        name="users", primary_key="id", data=pd.DataFrame(data={"id": [1, 2, 3]})
    )
    rel_data.add_table(
        name="events",
        primary_key="id",
        data=pd.DataFrame(data={"id": [1, 2, 3], "user_id": [1, 2, 3]}),
    )

    rel_data.add_foreign_key(foreign_key="events.user_id", referencing="users.id")
    assert len(rel_data.get_foreign_keys("events")) == 1

    rel_data.remove_foreign_key("events.user_id")
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


def test_get_subset_of_data(pets):
    normal_length = len(pets.get_table_data("humans"))
    subset = pets.get_table_data("humans", ["name", "city"])
    assert set(subset.columns) == {"name", "city"}
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


def test_table_with_json():
    bball_jsonl = """
    {"name": "LeBron James", "age": 38, "draft": {"year": 2003}, "rings": [{"team": "Heat", "year": 2012}, {"team": "Heat", "year": 2013}, {"team": "Cavaliers", "year": 2016}, {"team": "Lakers", "year": 2020}]}
    {"name": "Steph Curry", "age": 35, "draft": {"year": 2009, "college": "Davidson"}, "rings": [{"team": "Warriors", "year": 2015}, {"team": "Warriors", "year": 2017}, {"team": "Warriors", "year": 2018}, {"team": "Warriors", "year": 2022}]}
    """
    bball_df = pd.read_json(bball_jsonl, lines=True)
    rel_data = RelationalData()

    with patch("gretel_trainer.relational.json.make_suffix") as make_suffix:
        make_suffix.return_value = "sfx"
        rel_data.add_table(name="bball", primary_key="name", data=bball_df)

    # Ask for user-supplied tables (omit invented tables)
    assert set(rel_data.list_all_tables(Scope.PUBLIC)) == {"bball"}
    # We can optionally fetch all tables with scope="all"
    assert set(rel_data.list_all_tables(Scope.ALL)) == {
        "bball",
        "bball-sfx",
        "bball-rings-sfx",
    }
    # When asking for modelable tables, the user-supplied source is hidden
    # and the invented tables are included
    assert set(rel_data.list_all_tables(Scope.MODELABLE)) == {
        "bball-sfx",
        "bball-rings-sfx",
    }

    assert rel_data.get_foreign_keys("bball-rings-sfx") == [
        ForeignKey(
            table_name="bball-rings-sfx",
            columns=["bball~id"],
            parent_table_name="bball-sfx",
            parent_columns=["name"],
        )
    ]

    flattened_tables = {
        "bball-sfx": pd.DataFrame(
            data={
                "name": ["LeBron James", "Steph Curry"],
                "age": [38, 35],
                "draft>year": [2003, 2009],
                "draft>college": [None, "Davidson"],
            }
        ),
        "bball-rings-sfx": pd.DataFrame(
            data={
                "content>team": [
                    "Heat",
                    "Heat",
                    "Cavaliers",
                    "Lakers",
                    "Warriors",
                    "Warriors",
                    "Warriors",
                    "Warriors",
                ],
                "content>year": [2012, 2013, 2016, 2020, 2015, 2017, 2018, 2022],
                "~PRIMARY_KEY_ID~": [0, 1, 2, 3, 4, 5, 6, 7],
                "bball~id": [0, 0, 0, 0, 1, 1, 1, 1],
                "array~order": [0, 1, 2, 3, 0, 1, 2, 3],
            }
        ),
    }

    assert rel_data.get_table_columns("bball-sfx") == set(
        flattened_tables["bball-sfx"].columns
    )
    assert rel_data.get_table_columns("bball-rings-sfx") == set(
        flattened_tables["bball-rings-sfx"].columns
    )
    pdtest.assert_frame_equal(
        rel_data.get_table_data("bball-sfx"),
        flattened_tables["bball-sfx"],
        check_like=True,
    )
    pdtest.assert_frame_equal(
        rel_data.get_table_data("bball-rings-sfx"),
        flattened_tables["bball-rings-sfx"],
        check_like=True,
    )

    restored = rel_data.restore(flattened_tables)
    assert len(restored) == 1
    restored_bball_df = restored["bball"]
    pdtest.assert_frame_equal(
        restored_bball_df,
        pd.DataFrame(
            data={
                "name": ["LeBron James", "Steph Curry"],
                "age": [38, 35],
                "draft": [{"year": 2003}, {"year": 2009, "college": "Davidson"}],
                "rings": [
                    [
                        {"team": "Heat", "year": 2012},
                        {"team": "Heat", "year": 2013},
                        {"team": "Cavaliers", "year": 2016},
                        {"team": "Lakers", "year": 2020},
                    ],
                    [
                        {"team": "Warriors", "year": 2015},
                        {"team": "Warriors", "year": 2017},
                        {"team": "Warriors", "year": 2018},
                        {"team": "Warriors", "year": 2022},
                    ],
                ],
            }
        ),
    )
    pdtest.assert_frame_equal(restored_bball_df, bball_df)


def test_table_with_json_dict_only():
    bball_jsonl = """
    {"name": "LeBron James", "age": 38, "draft": {"year": 2003}}
    {"name": "Steph Curry", "age": 35, "draft": {"year": 2009, "college": "Davidson"}}
    """
    bball_df = pd.read_json(bball_jsonl, lines=True)
    rel_data = RelationalData()

    with patch("gretel_trainer.relational.json.make_suffix") as make_suffix:
        make_suffix.return_value = "sfx"
        rel_data.add_table(name="bball", primary_key="name", data=bball_df)

    assert set(rel_data.list_all_tables(Scope.MODELABLE)) == {"bball-sfx"}

    flattened_tables = {
        "bball-sfx": pd.DataFrame(
            data={
                "name": ["LeBron James", "Steph Curry"],
                "age": [38, 35],
                "draft>year": [2003, 2009],
                "draft>college": [None, "Davidson"],
            }
        )
    }

    assert rel_data.get_table_columns("bball-sfx") == set(
        flattened_tables["bball-sfx"].columns
    )
    pdtest.assert_frame_equal(
        rel_data.get_table_data("bball-sfx"),
        flattened_tables["bball-sfx"],
        check_like=True,
    )

    restored = rel_data.restore(flattened_tables)
    assert len(restored) == 1
    restored_bball_df = restored["bball"]
    pdtest.assert_frame_equal(
        restored_bball_df,
        pd.DataFrame(
            data={
                "name": ["LeBron James", "Steph Curry"],
                "age": [38, 35],
                "draft": [{"year": 2003}, {"year": 2009, "college": "Davidson"}],
            }
        ),
    )
    pdtest.assert_frame_equal(restored_bball_df, bball_df)


def test_more_json(documents):
    assert set(documents.list_all_tables(Scope.PUBLIC)) == {
        "users",
        "purchases",
        "payments",
    }

    assert set(documents.list_all_tables(Scope.ALL)) == {
        "users",
        "purchases",
        "payments",
        "purchases-sfx",
        "purchases-data-years-sfx",
    }

    assert set(documents.list_all_tables(Scope.MODELABLE)) == {
        "users",
        "payments",
        "purchases-sfx",
        "purchases-data-years-sfx",
    }

    assert set(documents.list_all_tables(Scope.EVALUATABLE)) == {
        "users",
        "payments",
        "purchases-sfx",
    }

    assert set(documents.get_table_columns("purchases-sfx")) == {
        "id",
        "user_id",
        "data>item",
        "data>cost",
        "data>details>color",
    }

    assert set(documents.get_table_columns("purchases-data-years-sfx")) == {
        "content",
        "~PRIMARY_KEY_ID~",
        "purchases~id",
        "array~order",
    }

    # Output tables MultiTable transforms or synthetics
    output_tables = {
        "users": pd.DataFrame(
            data={
                "id": [1, 2, 3],
                "name": ["Rob", "Sam", "Tim"],
            }
        ),
        "payments": pd.DataFrame(
            data={
                "id": [1, 2, 3, 4],
                "amount": [10, 10, 10, 10],
                "purchase_id": [1, 2, 3, 4],
            }
        ),
        "purchases-sfx": pd.DataFrame(
            data={
                "id": [1, 2, 3, 4],
                "user_id": [1, 1, 2, 3],
                "data>item": ["pen", "paint", "ink", "ink"],
                "data>cost": [18, 19, 20, 21],
                "data>details>color": ["blue", "yellow", "pink", "orange"],
            }
        ),
        "purchases-data-years-sfx": pd.DataFrame(
            data={
                "content": [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007],
                "~PRIMARY_KEY_ID~": [0, 1, 2, 3, 4, 5, 6, 7],
                "purchases~id": [0, 0, 0, 1, 2, 2, 3, 3],
                "array~order": [0, 1, 2, 0, 0, 1, 0, 1],
            }
        ),
    }

    restored_tables = documents.restore(output_tables)

    expected = {
        "users": output_tables["users"],
        "payments": output_tables["payments"],
        "purchases": pd.DataFrame(
            data={
                "id": [1, 2, 3, 4],
                "user_id": [1, 1, 2, 3],
                "data": [
                    {
                        "item": "pen",
                        "cost": 18,
                        "details": {"color": "blue"},
                        "years": [2000, 2001, 2002],
                    },
                    {
                        "item": "paint",
                        "cost": 19,
                        "details": {"color": "yellow"},
                        "years": [2003],
                    },
                    {
                        "item": "ink",
                        "cost": 20,
                        "details": {"color": "pink"},
                        "years": [2004, 2005],
                    },
                    {
                        "item": "ink",
                        "cost": 21,
                        "details": {"color": "orange"},
                        "years": [2006, 2007],
                    },
                ],
            }
        ),
    }

    for t, df in restored_tables.items():
        pdtest.assert_frame_equal(df, expected[t])

    assert documents.debug_summary() == {
        "foreign_key_count": 4,
        "max_depth": 2,
        "public_table_count": 3,
        "invented_table_count": 2,
        "tables": {
            "users": {
                "column_count": 2,
                "primary_key": ["id"],
                "foreign_key_count": 0,
                "foreign_keys": [],
                "is_invented_table": False,
            },
            "payments": {
                "column_count": 3,
                "primary_key": ["id"],
                "foreign_key_count": 1,
                "foreign_keys": [
                    {
                        "columns": ["purchase_id"],
                        "parent_table_name": "purchases-sfx",
                        "parent_columns": ["id"],
                    }
                ],
                "is_invented_table": False,
            },
            "purchases": {
                "column_count": 3,
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
            "purchases-sfx": {
                "column_count": 5,
                "primary_key": ["id"],
                "foreign_key_count": 1,
                "foreign_keys": [
                    {
                        "columns": ["user_id"],
                        "parent_table_name": "users",
                        "parent_columns": ["id"],
                    }
                ],
                "is_invented_table": True,
            },
            "purchases-data-years-sfx": {
                "column_count": 4,
                "primary_key": ["~PRIMARY_KEY_ID~"],
                "foreign_key_count": 1,
                "foreign_keys": [
                    {
                        "columns": ["purchases~id"],
                        "parent_table_name": "purchases-sfx",
                        "parent_columns": ["id"],
                    }
                ],
                "is_invented_table": True,
            },
        },
    }
