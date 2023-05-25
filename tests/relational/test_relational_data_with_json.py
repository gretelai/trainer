import pandas as pd
import pandas.testing as pdtest
import pytest

from gretel_trainer.relational.core import (
    ForeignKey,
    MultiTableException,
    RelationalData,
    Scope,
)
from gretel_trainer.relational.json import get_json_columns


@pytest.fixture
def bball():
    bball_jsonl = """
    {"name": "LeBron James", "age": 38, "draft": {"year": 2003}, "teams": ["Cavaliers", "Heat", "Lakers"]}
    {"name": "Steph Curry", "age": 35, "draft": {"year": 2009, "college": "Davidson"}, "teams": ["Warriors"]}
    """
    bball_df = pd.read_json(bball_jsonl, lines=True)

    rel_data = RelationalData()
    rel_data.add_table(name="bball", primary_key=None, data=bball_df)

    return rel_data


def test_list_json_cols(documents, bball):
    assert get_json_columns(documents.get_table_data("users")) == []
    assert get_json_columns(documents.get_table_data("purchases")) == ["data"]

    assert set(get_json_columns(bball.get_table_data("bball"))) == {"draft", "teams"}


def test_json_columns_produce_invented_flattened_tables(documents):
    pdtest.assert_frame_equal(
        documents.get_table_data("purchases-sfx"),
        pd.DataFrame(
            data={
                "~PRIMARY_KEY_ID~": [0, 1, 2, 3, 4, 5],
                "id": [1, 2, 3, 4, 5, 6],
                "user_id": [1, 2, 2, 3, 3, 3],
                "data>item": ["pen", "paint", "ink", "pen", "paint", "ink"],
                "data>cost": [100, 100, 100, 100, 100, 100],
                "data>details>color": ["red", "red", "red", "blue", "blue", "blue"],
            }
        ),
        check_like=True,
    )

    pdtest.assert_frame_equal(
        documents.get_table_data("purchases-data-years-sfx"),
        pd.DataFrame(
            data={
                "content": [2023, 2023, 2022, 2020, 2019, 2021],
                "array~order": [0, 0, 1, 0, 1, 0],
                "~PRIMARY_KEY_ID~": [0, 1, 2, 3, 4, 5],
                "purchases~id": [0, 1, 1, 2, 2, 4],
            }
        ),
        check_like=True,
        check_dtype=False,  # Without this, test fails asserting dtype mismatch in `content` field (object vs. int)
    )

    assert documents.get_foreign_keys("purchases-data-years-sfx") == [
        ForeignKey(
            table_name="purchases-data-years-sfx",
            columns=["purchases~id"],
            parent_table_name="purchases-sfx",
            parent_columns=["~PRIMARY_KEY_ID~"],
        )
    ]


def test_list_tables_accepts_various_scopes(documents):
    # PUBLIC reflects the user's source
    assert set(documents.list_all_tables(Scope.PUBLIC)) == {
        "users",
        "purchases",
        "payments",
    }

    # MODELABLE replaces any source tables containing JSON with the invented tables
    assert set(documents.list_all_tables(Scope.MODELABLE)) == {
        "users",
        "payments",
        "purchases-sfx",
        "purchases-data-years-sfx",
    }

    # EVALUATABLE is similar to MODELABLE, but omits invented child tables—we only evaluate the root invented table
    assert set(documents.list_all_tables(Scope.EVALUATABLE)) == {
        "users",
        "payments",
        "purchases-sfx",
    }

    # INVENTED returns only tables invented from source tables with JSON
    assert set(documents.list_all_tables(Scope.INVENTED)) == {
        "purchases-sfx",
        "purchases-data-years-sfx",
    }

    # ALL returns every table name, including both source-with-JSON tables and those invented from such tables
    assert set(documents.list_all_tables(Scope.ALL)) == {
        "users",
        "purchases",
        "payments",
        "purchases-sfx",
        "purchases-data-years-sfx",
    }

    # Default scope is MODELABLE
    assert set(documents.list_all_tables()) == set(
        documents.list_all_tables(Scope.MODELABLE)
    )


def test_get_modelable_table_names(documents):
    # Given a source-with-JSON name, returns the tables invented from that source
    assert set(documents.get_modelable_table_names("purchases")) == {
        "purchases-sfx",
        "purchases-data-years-sfx",
    }

    # Invented tables are modelable
    assert documents.get_modelable_table_names("purchases-sfx") == ["purchases-sfx"]
    assert documents.get_modelable_table_names("purchases-data-years-sfx") == [
        "purchases-data-years-sfx"
    ]

    # Unknown tables return empty list
    assert documents.get_modelable_table_names("nonsense") == []


def test_invented_json_column_names(documents, bball):
    # The root invented table adds columns for dictionary properties lifted from nested JSON objects
    assert set(documents.get_table_columns("purchases-sfx")) == {
        "~PRIMARY_KEY_ID~",
        "id",
        "user_id",
        "data>item",
        "data>cost",
        "data>details>color",
    }

    # JSON lists lead to invented child tables. These tables store the original content,
    # a new primary key, a foreign key back to the parent, and the original array index
    assert set(documents.get_table_columns("purchases-data-years-sfx")) == {
        "content",
        "~PRIMARY_KEY_ID~",
        "purchases~id",
        "array~order",
    }

    # If the source table does not have a primary key defined, one is created on the root invented table
    assert set(bball.get_table_columns("bball-sfx")) == {
        "name",
        "age",
        "draft>year",
        "draft>college",
        "~PRIMARY_KEY_ID~",
    }


def test_primary_key(documents, bball):
    # The root invented table's primary key is a composite key that includes the source PK and an invented column
    assert documents.get_primary_key("purchases") == ["id"]
    assert documents.get_primary_key("purchases-sfx") == ["id", "~PRIMARY_KEY_ID~"]

    assert bball.get_primary_key("bball") == []
    assert bball.get_primary_key("bball-sfx") == ["~PRIMARY_KEY_ID~"]

    # Setting an existing primary key to None puts us in the correct state
    assert len(documents.list_all_tables(Scope.ALL)) == 5
    original_payments_fks = documents.get_foreign_keys("payments")
    documents.set_primary_key(table="purchases", primary_key=None)
    assert len(documents.list_all_tables(Scope.ALL)) == 5
    assert documents.get_primary_key("purchases") == []
    assert documents.get_primary_key("purchases-sfx") == ["~PRIMARY_KEY_ID~"]
    assert documents.get_foreign_keys("purchases-data-years-sfx") == [
        ForeignKey(
            table_name="purchases-data-years-sfx",
            columns=["purchases~id"],
            parent_table_name="purchases-sfx",
            parent_columns=["~PRIMARY_KEY_ID~"],
        )
    ]
    assert documents.get_foreign_keys("payments") == original_payments_fks

    # Setting a None primary key to some column puts us in the correct state
    assert len(bball.list_all_tables(Scope.ALL)) == 3
    bball.set_primary_key(table="bball", primary_key="name")
    assert len(bball.list_all_tables(Scope.ALL)) == 3
    assert bball.get_primary_key("bball") == ["name"]
    assert bball.get_primary_key("bball-sfx") == ["name", "~PRIMARY_KEY_ID~"]
    assert bball.get_foreign_keys("bball-teams-sfx") == [
        ForeignKey(
            table_name="bball-teams-sfx",
            columns=["bball~id"],
            parent_table_name="bball-sfx",
            parent_columns=["~PRIMARY_KEY_ID~"],
        )
    ]


def test_foreign_keys(documents):
    # Foreign keys from the source-with-JSON table are present on the root invented table
    assert documents.get_foreign_keys("purchases") == documents.get_foreign_keys(
        "purchases-sfx"
    )

    # The root invented table name is used in the ForeignKey
    assert documents.get_foreign_keys("purchases") == [
        ForeignKey(
            table_name="purchases-sfx",
            columns=["user_id"],
            parent_table_name="users",
            parent_columns=["id"],
        )
    ]

    # Invented children point to invented parents
    assert documents.get_foreign_keys("purchases-data-years-sfx") == [
        ForeignKey(
            table_name="purchases-data-years-sfx",
            columns=["purchases~id"],
            parent_table_name="purchases-sfx",
            parent_columns=["~PRIMARY_KEY_ID~"],
        )
    ]

    # Source children of the source-with-JSON table point to the root invented table
    assert documents.get_foreign_keys("payments") == [
        ForeignKey(
            table_name="payments",
            columns=["purchase_id"],
            parent_table_name="purchases-sfx",
            parent_columns=["id"],
        )
    ]

    # You can request public/user-supplied names instead of the default invented table names
    assert documents.get_foreign_keys("payments", rename_invented_tables=True) == [
        ForeignKey(
            table_name="payments",
            columns=["purchase_id"],
            parent_table_name="purchases",
            parent_columns=["id"],
        )
    ]
    assert documents.get_foreign_keys("purchases", rename_invented_tables=True) == [
        ForeignKey(
            table_name="purchases",
            columns=["user_id"],
            parent_table_name="users",
            parent_columns=["id"],
        )
    ]

    # Removing a foreign key from the source-with-JSON table updates the root invented table
    documents.remove_foreign_key_constraint(
        table="purchases", constrained_columns=["user_id"]
    )
    assert documents.get_foreign_keys("purchases") == []
    assert documents.get_foreign_keys("purchases-sfx") == []


def test_update_data_with_existing_json_to_new_json(documents):
    new_purchases_jsonl = """
    {"id": 1, "user_id": 1, "data": {"item": "watercolor", "cost": 200, "details": {"color": "aquamarine"}, "years": [1999]}}
    {"id": 2, "user_id": 2, "data": {"item": "watercolor", "cost": 200, "details": {"color": "aquamarine"}, "years": [1999]}}
    {"id": 3, "user_id": 2, "data": {"item": "watercolor", "cost": 200, "details": {"color": "aquamarine"}, "years": [1999]}}
    {"id": 4, "user_id": 3, "data": {"item": "charcoal", "cost": 200, "details": {"color": "aquamarine"}, "years": [1998]}}
    {"id": 5, "user_id": 3, "data": {"item": "charcoal", "cost": 200, "details": {"color": "aquamarine"}, "years": [1998]}}
    {"id": 6, "user_id": 3, "data": {"item": "charcoal", "cost": 200, "details": {"color": "aquamarine"}, "years": [1998]}}
    """
    new_purchases_df = pd.read_json(new_purchases_jsonl, lines=True)

    documents.update_table_data("purchases", data=new_purchases_df)

    assert len(documents.list_all_tables(Scope.ALL)) == 5
    assert len(documents.list_all_tables(Scope.MODELABLE)) == 4

    expected = {
        "purchases-sfx": pd.DataFrame(
            data={
                "~PRIMARY_KEY_ID~": [0, 1, 2, 3, 4, 5],
                "id": [1, 2, 3, 4, 5, 6],
                "user_id": [1, 2, 2, 3, 3, 3],
                "data>item": [
                    "watercolor",
                    "watercolor",
                    "watercolor",
                    "charcoal",
                    "charcoal",
                    "charcoal",
                ],
                "data>cost": [200, 200, 200, 200, 200, 200],
                "data>details>color": [
                    "aquamarine",
                    "aquamarine",
                    "aquamarine",
                    "aquamarine",
                    "aquamarine",
                    "aquamarine",
                ],
            }
        ),
        "purchases-data-years-sfx": pd.DataFrame(
            data={
                "content": [1999, 1999, 1999, 1998, 1998, 1998],
                "array~order": [0, 0, 0, 0, 0, 0],
                "~PRIMARY_KEY_ID~": [0, 1, 2, 3, 4, 5],
                "purchases~id": [0, 1, 2, 3, 4, 5],
            }
        ),
    }

    pdtest.assert_frame_equal(
        documents.get_table_data("purchases-sfx"),
        expected["purchases-sfx"],
        check_like=True,
    )

    pdtest.assert_frame_equal(
        documents.get_table_data("purchases-data-years-sfx"),
        expected["purchases-data-years-sfx"],
        check_like=True,
        check_dtype=False,  # Without this, test fails asserting dtype mismatch in `content` field (object vs. int)
    )

    # User-supplied child table FK still exists
    assert documents.get_foreign_keys("payments") == [
        ForeignKey(
            table_name="payments",
            columns=["purchase_id"],
            parent_table_name="purchases-sfx",
            parent_columns=["id"],
        )
    ]


def test_update_data_existing_json_to_no_json(documents):
    new_purchases_df = pd.DataFrame(
        data={
            "id": [1, 2, 3, 4, 5, 6],
            "user_id": [1, 2, 2, 3, 3, 3],
            "data": ["pen", "paint", "ink", "pen", "paint", "ink"],
        }
    )

    documents.update_table_data("purchases", data=new_purchases_df)

    assert len(documents.list_all_tables(Scope.ALL)) == 3

    pdtest.assert_frame_equal(
        documents.get_table_data("purchases"),
        new_purchases_df,
        check_like=True,
    )

    assert documents.get_foreign_keys("payments") == [
        ForeignKey(
            table_name="payments",
            columns=["purchase_id"],
            parent_table_name="purchases",
            parent_columns=["id"],
        )
    ]


def test_update_data_existing_flat_to_json(documents):
    # Build up a RelationalData instance that basically mirrors documents,
    # but purchases is flat to start and thus there are no RelationalJson instances
    flat_purchases_df = pd.DataFrame(
        data={
            "id": [1, 2, 3, 4, 5, 6],
            "user_id": [1, 2, 2, 3, 3, 3],
            "data": ["pen", "paint", "ink", "pen", "paint", "ink"],
        }
    )
    rel_data = RelationalData()
    rel_data.add_table(
        name="users", primary_key="id", data=documents.get_table_data("users")
    )
    rel_data.add_table(name="purchases", primary_key="id", data=flat_purchases_df)
    rel_data.add_table(
        name="payments", primary_key="id", data=documents.get_table_data("payments")
    )
    rel_data.add_foreign_key_constraint(
        table="purchases",
        constrained_columns=["user_id"],
        referred_table="users",
        referred_columns=["id"],
    )
    rel_data.add_foreign_key_constraint(
        table="payments",
        constrained_columns=["purchase_id"],
        referred_table="purchases",
        referred_columns=["id"],
    )
    assert len(rel_data.list_all_tables(Scope.ALL)) == 3
    assert len(rel_data.list_all_tables(Scope.MODELABLE)) == 3

    rel_data.update_table_data("purchases", documents.get_table_data("purchases"))

    assert set(rel_data.list_all_tables(Scope.ALL)) == {
        "users",
        "purchases",
        "purchases-sfx",
        "purchases-data-years-sfx",
        "payments",
    }
    # the original purchases table is no longer flat, nor (therefore) MODELABLE
    assert set(rel_data.list_all_tables(Scope.MODELABLE)) == {
        "users",
        "purchases-sfx",
        "purchases-data-years-sfx",
        "payments",
    }
    assert rel_data.get_foreign_keys("payments") == [
        ForeignKey(
            table_name="payments",
            columns=["purchase_id"],
            parent_table_name="purchases-sfx",  # The foreign key now points to the root invented table
            parent_columns=["id"],
        )
    ]


# Simulates output tables from MultiTable transforms or synthetics, which will only include the MODELABLE tables
@pytest.fixture()
def mt_output_tables():
    return {
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
                "~PRIMARY_KEY_ID~": [0, 1, 2, 3],
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


def test_restoring_output_tables_to_original_shape(documents, mt_output_tables):
    restored_tables = documents.restore(mt_output_tables)

    # We expect our restored tables to match the PUBLIC tables
    assert len(restored_tables) == 3
    expected = {
        "users": mt_output_tables["users"],
        "payments": mt_output_tables["payments"],
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


def test_restore_with_incomplete_tableset(documents, mt_output_tables):
    without_invented_root = {
        k: v for k, v in mt_output_tables.items() if k != "purchases-sfx"
    }

    without_invented_child = {
        k: v for k, v in mt_output_tables.items() if k != "purchases-data-years-sfx"
    }

    restored_without_invented_root = documents.restore(without_invented_root)
    restored_without_invented_child = documents.restore(without_invented_child)

    # non-JSON-related tables are fine/unaffected
    pdtest.assert_frame_equal(
        restored_without_invented_child["users"], mt_output_tables["users"]
    )
    pdtest.assert_frame_equal(
        restored_without_invented_child["payments"], mt_output_tables["payments"]
    )
    pdtest.assert_frame_equal(
        restored_without_invented_root["users"], mt_output_tables["users"]
    )
    pdtest.assert_frame_equal(
        restored_without_invented_root["payments"], mt_output_tables["payments"]
    )

    # If the invented root is missing, the table is omitted from the result dict entirely
    assert "purchases" not in restored_without_invented_root

    # If an invented child is missing, we restore the shape but populate the list column with empty lists
    pdtest.assert_frame_equal(
        restored_without_invented_child["purchases"],
        pd.DataFrame(
            data={
                "id": [1, 2, 3, 4],
                "user_id": [1, 1, 2, 3],
                "data": [
                    {
                        "item": "pen",
                        "cost": 18,
                        "details": {"color": "blue"},
                        "years": [],
                    },
                    {
                        "item": "paint",
                        "cost": 19,
                        "details": {"color": "yellow"},
                        "years": [],
                    },
                    {
                        "item": "ink",
                        "cost": 20,
                        "details": {"color": "pink"},
                        "years": [],
                    },
                    {
                        "item": "ink",
                        "cost": 21,
                        "details": {"color": "orange"},
                        "years": [],
                    },
                ],
            }
        ),
    )


def test_flatten_and_restore_all_sorts_of_json():
    json = """
[
    {
        "a": 1,
        "b": {"bb": 1},
        "c": {"cc": {"ccc": 1}},
        "d": [1, 2, 3],
        "e": [
            {"ee": 1},
            {"ee": 2}
        ],
        "f": [
            {
                "ff": [
                    {"fff": 1},
                    {"fff": 2}
                ]
            }
        ],
    }
]
"""
    json_df = pd.read_json(json, orient="records")
    rel_data = RelationalData()
    rel_data.add_table(name="demo", primary_key=None, data=json_df)

    assert set(rel_data.list_all_tables(Scope.ALL)) == {
        "demo",
        "demo-sfx",
        "demo-d-sfx",
        "demo-e-sfx",
        "demo-f-sfx",
        "demo-f-content-ff-sfx",
    }

    assert rel_data.get_table_columns("demo-sfx") == {
        "a",
        "b>bb",
        "c>cc>ccc",
        "~PRIMARY_KEY_ID~",
    }
    assert rel_data.get_table_columns("demo-d-sfx") == {
        "content",
        "~PRIMARY_KEY_ID~",
        "demo~id",
        "array~order",
    }
    assert rel_data.get_table_columns("demo-e-sfx") == {
        "content>ee",
        "~PRIMARY_KEY_ID~",
        "demo~id",
        "array~order",
    }
    assert rel_data.get_table_columns("demo-f-sfx") == {
        "~PRIMARY_KEY_ID~",
        "demo~id",
        "array~order",
    }
    assert rel_data.get_table_columns("demo-f-content-ff-sfx") == {
        "content>fff",
        "~PRIMARY_KEY_ID~",
        "demo^f~id",
        "array~order",
    }

    output_tables = {
        "demo-sfx": pd.DataFrame(
            data={
                "a": [1, 2],
                "b>bb": [3, 4],
                "c>cc>ccc": [5, 6],
                "~PRIMARY_KEY_ID~": [0, 1],
            }
        ),
        "demo-d-sfx": pd.DataFrame(
            data={
                "content": [10, 11, 12, 13],
                "~PRIMARY_KEY_ID~": [0, 1, 2, 3],
                "demo~id": [0, 0, 0, 1],
                "array~order": [0, 1, 2, 0],
            }
        ),
        "demo-e-sfx": pd.DataFrame(
            data={
                "content>ee": [100, 200, 300],
                "~PRIMARY_KEY_ID~": [0, 1, 2],
                "demo~id": [0, 1, 1],
                "array~order": [0, 0, 1],
            }
        ),
        "demo-f-sfx": pd.DataFrame(
            data={"~PRIMARY_KEY_ID~": [0, 1], "demo~id": [0, 1], "array~order": [0, 0]}
        ),
        "demo-f-content-ff-sfx": pd.DataFrame(
            data={
                "content>fff": [10, 11, 12],
                "~PRIMARY_KEY_ID~": [0, 1, 2],
                "demo^f~id": [0, 0, 0],
                "array~order": [0, 1, 2],
            }
        ),
    }

    restored = rel_data.restore(output_tables)

    expected = pd.DataFrame(
        data={
            "a": [1, 2],
            "b": [{"bb": 3}, {"bb": 4}],
            "c": [{"cc": {"ccc": 5}}, {"cc": {"ccc": 6}}],
            "d": [[10, 11, 12], [13]],
            "e": [[{"ee": 100}], [{"ee": 200}, {"ee": 300}]],
            "f": [[{"ff": [{"fff": 10}, {"fff": 11}, {"fff": 12}]}], [{"ff": []}]],
        }
    )

    assert len(restored) == 1
    pdtest.assert_frame_equal(restored["demo"], expected)


def test_only_lists_edge_case():
    # Smallest reproduction: a dataframe with just one row and one column, and the value is a list
    list_df = pd.DataFrame(data={"l": [[1, 2, 3, 4]]})
    rel_data = RelationalData()

    # Since there are no flat fields on the source, we cannot create an invented root table.
    # Internally, the call below fails when trying to create a foreign key from the invented child table up to a nonexistent root.
    with pytest.raises(MultiTableException):
        rel_data.add_table(name="list", primary_key=None, data=list_df)

    # TODO: Ideally we should "roll back the transaction" and the call below should return an empty set,
    # but instead we currently get left in a nonsensical state.
    assert set(rel_data.list_all_tables(Scope.ALL)) == {"list", "list-l-sfx"}


def test_lists_of_lists():
    # Enough flat data in the source to create a root invented table.
    # Upping the complexity by making the special value a list of lists,
    # but not to fear: we can handle this correctly.
    lol_df = pd.DataFrame(data={"a": [1], "l": [[[1, 2], [3, 4]]]})
    rel_data = RelationalData()
    rel_data.add_table(name="lol", primary_key=None, data=lol_df)

    assert set(rel_data.list_all_tables(Scope.ALL)) == {
        "lol",
        "lol-sfx",
        "lol-l-sfx",
        "lol-l-content-sfx",
    }

    output = {
        "lol-sfx": pd.DataFrame(data={"a": [1, 2], "~PRIMARY_KEY_ID~": [0, 1]}),
        "lol-l-sfx": pd.DataFrame(
            data={"~PRIMARY_KEY_ID~": [0, 1], "lol~id": [0, 0], "array~order": [0, 1]}
        ),
        "lol-l-content-sfx": pd.DataFrame(
            data={
                "content": [10, 20, 30, 40],
                "~PRIMARY_KEY_ID~": [0, 1, 2, 3],
                "lol^l~id": [0, 0, 1, 1],
                "array~order": [0, 1, 0, 1],
            }
        ),
    }
    restored = rel_data.restore(output)

    assert len(restored) == 1
    pdtest.assert_frame_equal(
        restored["lol"],
        pd.DataFrame(
            data={
                "a": [1, 2],
                "l": [[[10, 20], [30, 40]], []],
            }
        ),
    )


def test_mix_of_dict_and_list_cols():
    df = pd.DataFrame(
        data={
            "id": [1, 2],
            "dcol": [{"language": "english"}, {"language": "spanish"}],
            "lcol": [["a", "b"], ["c", "d"]],
        }
    )
    rel_data = RelationalData()
    rel_data.add_table(name="mix", primary_key=None, data=df)
    assert set(rel_data.list_all_tables()) == {
        "mix-sfx",
        "mix-lcol-sfx",
    }
    assert set(rel_data.get_table_data("mix-sfx").columns) == {
        "id",
        "~PRIMARY_KEY_ID~",
        "dcol>language",
    }
    assert set(rel_data.get_table_data("mix-lcol-sfx").columns) == {
        "~PRIMARY_KEY_ID~",
        "content",
        "array~order",
        "mix~id",
    }


def test_all_tables_are_present_in_debug_summary(documents):
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
                "column_count": 6,
                "primary_key": ["id", "~PRIMARY_KEY_ID~"],
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
                        "parent_columns": ["~PRIMARY_KEY_ID~"],
                    }
                ],
                "is_invented_table": True,
            },
        },
    }
