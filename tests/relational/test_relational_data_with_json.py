from unittest.mock import patch

import pandas as pd
import pandas.testing as pdtest
import pytest

from gretel_trainer.relational.core import ForeignKey, RelationalData, Scope


@pytest.fixture
def bball():
    bball_jsonl = """
    {"name": "LeBron James", "age": 38, "draft": {"year": 2003}}
    {"name": "Steph Curry", "age": 35, "draft": {"year": 2009, "college": "Davidson"}}
    """
    bball_df = pd.read_json(bball_jsonl, lines=True)

    rel_data = RelationalData()
    with patch("gretel_trainer.relational.json.make_suffix") as make_suffix:
        make_suffix.return_value = "sfx"
        rel_data.add_table(name="bball", primary_key=None, data=bball_df)

    return rel_data


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


def test_invented_json_column_names(documents, bball):
    # The root invented table adds columns for dictionary properties lifted from nested JSON objects
    assert set(documents.get_table_columns("purchases-sfx")) == {
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
    # Typically, the source-with-JSON and root invented tables' primary keys are identical
    assert documents.get_primary_key("purchases") == documents.get_primary_key(
        "purchases-sfx"
    )

    # This is not the case if the source-with-JSON does not have a primary key
    assert bball.get_primary_key("bball") == []
    assert bball.get_primary_key("bball-sfx") == ["~PRIMARY_KEY_ID~"]


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
            parent_columns=["id"],
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


def test_restoring_output_tables_to_original_shape(documents):
    # Output tables from MultiTable transforms or synthetics will include only the MODELABLE tables
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

    # We expect our restored tables to match the PUBLIC tables
    assert len(restored_tables) == 3
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
