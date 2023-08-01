from pathlib import Path
from typing import List

import pandas as pd
import pytest

from gretel_trainer.strategy import PartitionConstraints, PartitionStrategy


@pytest.fixture(scope="module", autouse=True)
def test_df() -> pd.DataFrame:
    return pd.read_csv(str(Path(__file__).parent / "data" / "core-221-train.csv"))


@pytest.fixture(scope="module")
def test_seeds() -> List[str]:
    return ["goal", "goal_type", "goals"]


@pytest.mark.parametrize(
    "constraints",
    [
        PartitionConstraints(max_row_count=1000),
        PartitionConstraints(max_row_count=100),
    ],
)
def test_strategy_all_columns(constraints: PartitionConstraints, test_df):
    strategy = PartitionStrategy.from_dataframe("foo", test_df, constraints)
    assert (
        len(test_df) // constraints.max_row_count
        <= strategy.partition_count
        <= len(test_df) // constraints.max_row_count + 1
    )

    # partitions are of roughly equal size
    extracted_df_lengths = [len(partition.extract_df(test_df)) for partition in strategy.partitions]
    assert max(extracted_df_lengths) - min(extracted_df_lengths) <= 1

    # re-assemble all partitions and compare
    compare = pd.DataFrame()
    for idx, partition in enumerate(strategy.partitions):
        compare = pd.concat([compare, partition.extract_df(test_df)]).reset_index(
            drop=True
        )
        assert idx == partition.idx

    assert compare.shape == test_df.shape


def test_strategy_seeds(test_df, test_seeds):
    constraints = PartitionConstraints(max_row_count=100)
    constraints.seed_headers = test_seeds
    strategy = PartitionStrategy.from_dataframe("foo", test_df, constraints)
    for partition in strategy.partitions:
        assert partition.columns.seed_headers == test_seeds


def test_read_write(test_df, tmpdir):
    save_location = Path(tmpdir) / "data.json"
    constraints = PartitionConstraints(max_row_count=100)
    strategy = PartitionStrategy.from_dataframe("foo", test_df, constraints)

    # Inproper filename
    with pytest.raises(ValueError):
        strategy.save_to("foo.txt")

    # Actually save
    strategy.save_to(save_location)

    # Can't overwrite
    with pytest.raises(RuntimeError):
        strategy.save_to(save_location)

    # Overwrite
    strategy.save_to(save_location, overwrite=True)

    # Load from disk and make sure we have the same obj
    strategy_copy = PartitionStrategy.from_disk(save_location)
    assert strategy_copy.dict() == strategy.dict()
    # Make sure we can save
    strategy_copy.save()

    # All partitions have no ctx yet
    assert len(strategy.partitions_no_ctx) == strategy.partition_count

    # Update one partition and query it
    strategy.update_partition(0, {"model_id": "abc123"})
    assert len(strategy.partitions_no_ctx) == strategy.partition_count - 1
    check_partition = strategy.query_partitions({"model_id": "abc123"})
    assert len(check_partition) == 1
    assert check_partition[0].ctx == {"model_id": "abc123"}

    # Query for multiple partitions
    strategy.update_partition(0, {"key": "findme"})
    strategy.update_partition(1, {"key": "findme"})
    assert len(strategy.partitions_no_ctx) == strategy.partition_count - 2
    results = strategy.query_partitions({"key": "findme"})
    assert len(results) == 2
    assert results[0].ctx == {"model_id": "abc123", "key": "findme"}
    assert results[1].ctx == {"key": "findme"}

    # Empty query is a "find all"
    assert len(strategy.query_partitions({})) == strategy.partition_count

    # No matching hits
    assert not strategy.query_partitions({"nope": "nope"})

    # Do a manual update of a partition and query
    partition = strategy.partitions_no_ctx[0]
    partition.update_ctx({"foo": "bar"})
    partition.update_ctx({"model_id": "anotherid"})
    assert len(strategy.query_partitions({"foo": "bar"})) == 1

    # Do some glob queries
    assert len(strategy.query_glob("key", "find*")) == 2
    assert len(strategy.query_glob("model_id", "*")) == 2
