import pytest

from gretel_trainer.relational.connectors import sqlite_conn
from gretel_trainer.relational.core import MultiTableException


def test_extracting_relational_data(local_db, ecom):
    sqlite = sqlite_conn(local_db)
    extracted = sqlite.extract()

    all_tables = extracted.list_all_tables()
    assert set(all_tables) == {
        "users",
        "events",
        "products",
        "distribution_center",
        "order_items",
        "inventory_items",
    }

    for table in all_tables:
        assert extracted.get_parents(table) == ecom.get_parents(table)
        assert extracted.get_foreign_keys(table) == ecom.get_foreign_keys(table)


def test_extract_subsets_of_relational_data(local_db):
    sqlite = sqlite_conn(local_db)

    with pytest.raises(MultiTableException):
        sqlite.extract(only=["users"], ignore=["events"])

    only = sqlite.extract(only=["users", "events", "products"])
    ignore = sqlite.extract(
        ignore=["distribution_center", "order_items", "inventory_items"]
    )

    expected_tables = {"users", "events", "products"}
    assert set(only.list_all_tables()) == expected_tables
    assert set(ignore.list_all_tables()) == expected_tables

    # `products` has a foreign key to `distribution_center` in the source, but because the
    # latter table was not extracted, the relationship is not recognized
    assert only.get_parents("products") == []
    assert ignore.get_parents("products") == []
