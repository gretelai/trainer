from unittest.mock import patch

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

    with patch("random.shuffle") as shuffle:
        shuffle = sorted
        result = common.make_composite_pk_columns(
            table_name="table",
            rel_data=rel_data,
            primary_key=["letter", "number"],
            synth_row_count=8,
            record_size_ratio=1.0,
        )

    assert result == [
        (0, 0, 0, 0, 1, 1, 1, 1),
        (0, 1, 2, 3, 0, 1, 2, 3),
    ]
