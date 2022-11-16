from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

class MultiTableException(Exception):
    pass


@dataclass
class PrimaryKey:
    table_name: str
    column_name: str


@dataclass
class ForeignKey:
    table_name: str
    column_name: str
    parent_table_name: str
    parent_column_name: str


@dataclass
class Table:
    name: str
    data: pd.DataFrame
    path: Path
    primary_key: Optional[PrimaryKey]


# rdb_config: Dict[str, Any]
#   table_data: Dict[str, pd.DataFrame]
#   table_files: Dict[str, str]
#   primary_keys: Dict[str, str]
#   relationships: List[List[Tuple[str, str]]]


@dataclass
class Source:
    """
    Raw data and metadata describing the source relational data.

    Stand-in connectors written in Python may produce instances of this object directly.

    More broadly, this type represents the contract between the MultiTable model
    and connectors written in *any* language. Connectors must be capable of
    producing a JSON document that can be parsed into this object (see `from_metadata`);
    for instance, it should include pointers to CSV files containing exported
    source table data, as well as a representation of table relationships.
    """
    tables: Dict[str, Table]
    # relationships_typed: Dict[PrimaryKey, List[ForeignKey]] # Not sure yet what this actually should look like

    @classmethod
    def from_metadata(cls, metadata_path: str) -> Source: # type: ignore (deferring implementation)
        """
        Construct a Source instance using the JSON metadata located at `metadata_path`.
        """
        pass


@dataclass
class SyntheticTables:
    """
    Holds the synthetic data produced by the MultiTable model.
    Keys are source table names, and values are pandas DataFrames.

    Stand-in connectors written in Python may consume instances of this object directly.

    More broadly, this type represents the contract between the MultiTable model
    and connectors written in *any* language. The MultiTable model will write this object to
    a directory (see `export_to_filesystem`) for a sink connector to ingest and persist.
    """
    data: Dict[str, pd.DataFrame]

    def export_to_filesystem(self, out_dir: str) -> Path: # type: ignore (deferring implementation)
        """
        Write each table as a CSV, plus one JSON metadata file, to `out_dir`.
        Returns the metadata file path.
        """
        pass
