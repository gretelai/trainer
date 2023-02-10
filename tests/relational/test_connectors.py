import sqlite3
import tempfile

import pytest

from gretel_trainer.relational.connectors import sqlite_conn
from gretel_trainer.relational.core import MultiTableException


def test_extract_subsets_of_relational_data(example_dbs):
    with tempfile.NamedTemporaryFile() as f:
        con = sqlite3.connect(f.name)
        cur = con.cursor()
        with open(example_dbs / "ecom.sql") as sql_script:
            cur.executescript(sql_script.read())

        connector = sqlite_conn(f.name)

        with pytest.raises(MultiTableException):
            connector.extract(only=["users"], ignore=["events"])

        only = connector.extract(only=["users", "events", "products"])
        ignore = connector.extract(
            ignore=["distribution_center", "order_items", "inventory_items"]
        )

    expected_tables = {"users", "events", "products"}
    assert set(only.list_all_tables()) == expected_tables
    assert set(ignore.list_all_tables()) == expected_tables

    # `products` has a foreign key to `distribution_center` in the source, but because the
    # latter table was not extracted, the relationship is not recognized
    assert only.get_parents("products") == []
    assert ignore.get_parents("products") == []
