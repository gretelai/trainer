from pathlib import Path
from typing import List
from dataclasses import dataclass

import pandas as pd
import pytest

from gretel_synthetics.utils.header_clusters import cluster
from gretel_trainer.strategy import PartitionConstraints, PartitionStrategy


@pytest.fixture(scope="module", autouse=True)
def test_df() -> pd.DataFrame:
    return pd.read_csv(str(Path(__file__).parent / "data" / "core-221-train.csv"))


@pytest.fixture(scope="module")
def header_clusters(test_df) -> List[List[str]]:
    clusters = cluster(test_df)
    assert len(clusters) == 2
    return clusters


@dataclass
class ClusterData:
    clusters: List[List[str]]
    seeds: List[str]


@pytest.fixture(scope="module")
def header_clusters_seed(test_df) -> ClusterData:
    seeds = ["goal", "goal_type", "goals"]
    clusters = cluster(test_df, header_prefix=seeds)
    return ClusterData(clusters=clusters, seeds=seeds)


def test_invalid_partition_constraints():
    with pytest.raises(AttributeError):
        PartitionConstraints(max_row_count=1, max_row_partitions=1)

    with pytest.raises(AttributeError):
        PartitionConstraints()


@pytest.mark.parametrize(
    "constraints",
    [
        PartitionConstraints(max_row_partitions=3),
        PartitionConstraints(max_row_partitions=20),
        PartitionConstraints(max_row_count=1000),
        PartitionConstraints(max_row_count=100),
    ],
)
def test_strategy_all_columns(constraints: PartitionConstraints, test_df):
    strategy = PartitionStrategy.from_dataframe("foo", test_df, constraints)
    if constraints.max_row_partitions:
        assert constraints.max_row_partitions == strategy.partition_count

    if constraints.max_row_count:
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


@pytest.mark.parametrize(
    "constraints",
    [
        PartitionConstraints(max_row_partitions=3),
        PartitionConstraints(max_row_partitions=20),
        PartitionConstraints(max_row_count=1000),
        PartitionConstraints(max_row_count=100),
    ],
)
def test_strategy_column_batches(
    constraints: PartitionConstraints, test_df, header_clusters
):
    constraints.header_clusters = header_clusters

    strategy = PartitionStrategy.from_dataframe("foo", test_df, constraints)
    if constraints.max_row_partitions:
        assert (
            constraints.max_row_partitions * len(header_clusters)
            == strategy.partition_count
        )

    if constraints.max_row_count:
        assert (
            len(test_df) // constraints.max_row_count
            <= strategy.partition_count / len(header_clusters)
            <= len(test_df) // constraints.max_row_count + 1
        )

    # partitions are of roughly equal size
    extracted_df_lengths = [len(partition.extract_df(test_df)) for partition in strategy.partitions]
    assert max(extracted_df_lengths) - min(extracted_df_lengths) <= 1

    part1 = pd.DataFrame()
    part2 = pd.DataFrame()
    for idx, partition in enumerate(strategy.partitions):
        assert partition.idx == idx
        tmp_df = partition.extract_df(test_df)
        if list(tmp_df.columns) == header_clusters[0]:
            part1 = pd.concat([part1, tmp_df]).reset_index(drop=True)
        else:
            part2 = pd.concat([part2, tmp_df]).reset_index(drop=True)

    final = pd.concat([part1, part2], axis=1)
    assert final.shape == test_df.shape


@pytest.mark.parametrize(
    "constraints",
    [
        PartitionConstraints(max_row_partitions=3),
        PartitionConstraints(max_row_count=100),
    ],
)
def test_strategy_seeds(constraints: PartitionConstraints, test_df, header_clusters_seed: ClusterData):
    constraints.header_clusters = header_clusters_seed.clusters
    constraints.seed_headers = header_clusters_seed.seeds
    strategy = PartitionStrategy.from_dataframe("foo", test_df, constraints)
    for partition in strategy.partitions:
        if partition.columns.idx == 0:
            assert partition.columns.seed_headers == header_clusters_seed.seeds
        else:
            assert not partition.columns.seed_headers


def test_read_write(test_df, header_clusters, tmpdir):
    save_location = Path(tmpdir) / "data.json"
    constraints = PartitionConstraints(
        max_row_partitions=3, header_clusters=header_clusters
    )
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
