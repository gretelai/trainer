import pandas as pd

import gretel_trainer.relational.strategies.common as common
from gretel_trainer.relational.core import RelationalData


def test_composite_pk_columns():
    table = pd.DataFrame(
        data={
            "letter": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "number": [1, 2, 3, 4, 1, 2, 3, 4],
        }
    )
    rel_data = RelationalData()
    rel_data.add_table(
        name="table",
        primary_key=["letter", "number"],
        data=table,
    )

    result = common.make_composite_pk_columns(
        table_name="table",
        rel_data=rel_data,
        primary_key=["letter", "number"],
        synth_row_count=8,
        record_size_ratio=1.0,
    )

    # There is a tuple of values for each primary key column
    assert len(result) == 2

    # Each tuple has enough values for the synthetic result
    for t in result:
        assert len(t) == 8

    # Each combination is unique
    synthetic_pks = set(zip(*result))
    assert len(synthetic_pks) == 8

    # The set of unique values in each synthetic column roughly matches
    # the set of unique values in the source columns.
    # In this example they match exactly because there are no other possible combinations,
    # but in practice it's possible to randomly not-select some values.
    assert len(set(result[0])) == 2
    assert len(set(result[1])) == 4


def test_composite_pk_columns_2():
    table = pd.DataFrame(
        data={
            "letter": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "number": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    rel_data = RelationalData()
    rel_data.add_table(
        name="table",
        primary_key=["letter", "number"],
        data=table,
    )

    result = common.make_composite_pk_columns(
        table_name="table",
        rel_data=rel_data,
        primary_key=["letter", "number"],
        synth_row_count=8,
        record_size_ratio=1.0,
    )

    # There is a tuple of values for each primary key column
    assert len(result) == 2

    # Each tuple has enough values for the synthetic result
    for t in result:
        assert len(t) == 8

    # Each combination is unique
    synthetic_pks = set(zip(*result))
    assert len(synthetic_pks) == 8

    # The set of unique values in each synthetic column roughly matches
    # the set of unique values in the source columns.
    # In this example, there are more potential combinations than there are synthetic rows,
    # so our assertions are not as strict.
    assert len(set(result[0])) <= 2
    assert len(set(result[1])) <= 8
