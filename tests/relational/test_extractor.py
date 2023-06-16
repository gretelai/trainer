import sqlite3
import tempfile
from pathlib import Path
from typing import Iterable

import pytest

from gretel_trainer.relational.connectors import Connector, sqlite_conn
from gretel_trainer.relational.extractor import (
    ExtractorConfig,
    TableExtractor,
    _determine_sample_size,
)


def test_subset_config():
    # Can't have a target row count < -1
    with pytest.raises(ValueError):
        ExtractorConfig(target_row_count=-2)

    # Concrete row count
    config = ExtractorConfig(target_row_count=100)
    assert _determine_sample_size(config, 200) == 100

    # Ratio
    config = ExtractorConfig(target_row_count=0.5)
    assert _determine_sample_size(config, 100) == 50

    # Entire table
    config = ExtractorConfig()
    assert config.entire_table
    assert _determine_sample_size(config, 101) == 101

    # Empty table
    config = ExtractorConfig(target_row_count=0)
    assert not config.entire_table
    assert config.empty_table
    assert _determine_sample_size(config, 101) == 0

    # Can't have both only and ignore
    with pytest.raises(ValueError):
        ExtractorConfig(ignore={"foo"}, only={"bar"})


@pytest.fixture
def connector_ecom(example_dbs) -> Iterable[Connector]:
    with tempfile.NamedTemporaryFile() as f:
        con = sqlite3.connect(f.name)
        cur = con.cursor()
        with open(example_dbs / "ecom.sql") as sql_script:
            cur.executescript(sql_script.read())

        connector = sqlite_conn(f.name)
        yield connector


@pytest.fixture
def connector_art(example_dbs) -> Iterable[Connector]:
    with tempfile.NamedTemporaryFile() as f:
        con = sqlite3.connect(f.name)
        cur = con.cursor()
        with open(example_dbs / "art.sql") as sql_script:
            cur.executescript(sql_script.read())

        connector = sqlite_conn(f.name)
        yield connector


def test_extract_schema(connector_ecom: Connector, tmpdir):
    config = ExtractorConfig()
    extractor = TableExtractor(
        config=config, connector=connector_ecom, storage_dir=Path(tmpdir)
    )
    extractor._extract_schema()
    assert extractor.table_order == [
        "events",
        "order_items",
        "users",
        "inventory_items",
        "products",
        "distribution_center",
    ]


def test_table_session(connector_art, tmpdir):
    config = ExtractorConfig()
    extractor = TableExtractor(
        config=config, connector=connector_art, storage_dir=Path(tmpdir)
    )
    table_session = extractor._get_table_session("paintings")
    assert table_session.total_row_count == 7
    assert set(table_session.columns) == {"id", "artist_id", "name"}
    assert table_session.total_column_count == 3


@pytest.mark.parametrize("target,expect", [(-1, 7), (3, 3), (0, 0)])
def test_sample_table(target, expect, connector_art, tmpdir):
    config = ExtractorConfig(target_row_count=target)
    extractor = TableExtractor(
        config=config, connector=connector_art, storage_dir=Path(tmpdir)
    )
    extractor._chunk_size = 1
    meta = extractor._sample_table("paintings")
    assert meta.original_row_count == 7
    assert meta.sampled_row_count == expect
    assert meta.column_count == 3
    df = extractor.get_table_df("paintings")
    assert len(df) == expect

    # Now we can sample from an intermediate table
    meta = extractor._sample_table("artists", child_tables=["paintings"])
    assert meta.original_row_count == 4
    assert (
        0 <= meta.sampled_row_count <= 4
    )  # could vary based on what other FKs were selected
    df = extractor.get_table_df("artists")
    assert 0 <= len(df) <= 4

    # A001 should never be sampled, sorry Wassily
    assert "A001" not in df["id"]


@pytest.mark.parametrize("sample_mode", ["random", "contiguous"])
def test_sample_tables(sample_mode, connector_art, tmpdir):
    config = ExtractorConfig(target_row_count=0.5, sample_mode=sample_mode)
    extractor = TableExtractor(
        config=config,
        connector=connector_art,
        storage_dir=Path(tmpdir),
    )
    meta = extractor.sample_tables()
    paintings = meta["paintings"]
    assert paintings.sampled_row_count == 3

    # artists will vary between 1 and 3
    artists = meta["artists"]
    assert 1 <= artists.sampled_row_count <= 3
