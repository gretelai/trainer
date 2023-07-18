import pandas as pd

import gretel_trainer.relational.strategies.common as common
from gretel_trainer.relational.core import RelationalData


def test_composite_pk_columns(tmpdir):
    df = pd.DataFrame(
        data={
            "letter": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "number": [1, 2, 3, 4, 1, 2, 3, 4],
        }
    )
    rel_data = RelationalData(directory=tmpdir)
    rel_data.add_table(
        name="table",
        primary_key=["letter", "number"],
        data=df,
    )

    result = common.make_composite_pks(
        table_name="table",
        rel_data=rel_data,
        primary_key=["letter", "number"],
        synth_row_count=8,
    )

    # Label-encoding turns the keys into zero-indexed contiguous integers.
    # It is absolutely required that all composite keys returned are unique.
    # We also ideally recreate the original data frequencies (in this case,
    # two unique letters and four unique numbers).
    expected_keys = [
        {"letter": 0, "number": 0},
        {"letter": 0, "number": 1},
        {"letter": 0, "number": 2},
        {"letter": 0, "number": 3},
        {"letter": 1, "number": 0},
        {"letter": 1, "number": 1},
        {"letter": 1, "number": 2},
        {"letter": 1, "number": 3},
    ]

    for expected_key in expected_keys:
        assert expected_key in result


def test_composite_pk_columns_2(tmpdir):
    df = pd.DataFrame(
        data={
            "letter": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "number": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    rel_data = RelationalData(directory=tmpdir)
    rel_data.add_table(
        name="table",
        primary_key=["letter", "number"],
        data=df,
    )

    result = common.make_composite_pks(
        table_name="table",
        rel_data=rel_data,
        primary_key=["letter", "number"],
        synth_row_count=8,
    )

    # We create as many keys as we need
    assert len(result) == 8

    # Each combination is unique
    assert len(set([str(composite_key) for composite_key in result])) == 8

    # In this case, there are more potential unique combinations than there are synthetic rows,
    # so we can't say for sure what the exact composite values will be. However, we do expect
    # the original frequencies to be maintained.
    synthetic_letters = [key["letter"] for key in result]
    assert len(synthetic_letters) == 8
    assert set(synthetic_letters) == {0, 1}
    assert len([x for x in synthetic_letters if x != 0]) == 4

    synthetic_numbers = [key["number"] for key in result]
    assert len(synthetic_numbers) == 8
    assert set(synthetic_numbers) == {0, 1, 2, 3, 4, 5, 6, 7}
