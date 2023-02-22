from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from gretel_trainer.relational.artifacts import ArtifactCollection
from gretel_trainer.relational.core import ForeignKey, RelationalData


@dataclass
class BackupRelationalDataTable:
    primary_key: Optional[str]


@dataclass
class BackupForeignKey:
    foreign_key: str
    referencing: str

    @classmethod
    def from_fk(cls, fk: ForeignKey) -> BackupForeignKey:
        return BackupForeignKey(
            foreign_key=f"{fk.table_name}.{fk.column_name}",
            referencing=f"{fk.parent_table_name}.{fk.parent_column_name}",
        )


@dataclass
class BackupRelationalData:
    tables: Dict[str, BackupRelationalDataTable]
    foreign_keys: List[BackupForeignKey]

    @classmethod
    def from_relational_data(cls, rel_data: RelationalData) -> BackupRelationalData:
        tables = {}
        foreign_keys = []
        for table in rel_data.list_all_tables():
            tables[table] = BackupRelationalDataTable(
                primary_key=rel_data.get_primary_key(table),
            )
            foreign_keys.extend(
                [
                    BackupForeignKey.from_fk(key)
                    for key in rel_data.get_foreign_keys(table)
                ]
            )
        return BackupRelationalData(tables=tables, foreign_keys=foreign_keys)


@dataclass
class BackupTrainTable:
    model_id: str
    training_columns: List[str]


@dataclass
class BackupTransforms:
    model_ids: Dict[str, str]


@dataclass
class BackupTrain:
    tables: Dict[str, BackupTrainTable]


@dataclass
class BackupGenerate:
    identifier: str
    preserved: List[str]
    record_size_ratio: float
    record_handler_ids: Dict[str, str]
    lost_contact: List[str]
    missing_model: List[str]


@dataclass
class Backup:
    project_name: str
    strategy: str
    gretel_model: str
    working_dir: str
    refresh_interval: int
    artifact_collection: ArtifactCollection
    relational_data: BackupRelationalData
    transforms: Optional[BackupTransforms] = None
    train: Optional[BackupTrain] = None
    generate: Optional[BackupGenerate] = None

    @property
    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, b: Dict[str, Any]):
        relational_data = b["relational_data"]
        brd = BackupRelationalData(
            tables={
                k: BackupRelationalDataTable(**v)
                for k, v in relational_data.get("tables", {}).items()
            },
            foreign_keys=[
                BackupForeignKey(
                    foreign_key=fk["foreign_key"],
                    referencing=fk["referencing"],
                )
                for fk in relational_data.get("foreign_keys", [])
            ],
        )

        backup = Backup(
            project_name=b["project_name"],
            strategy=b["strategy"],
            gretel_model=b["gretel_model"],
            working_dir=b["working_dir"],
            refresh_interval=b["refresh_interval"],
            artifact_collection=ArtifactCollection(**b["artifact_collection"]),
            relational_data=brd,
        )

        transforms = b.get("transforms")
        if transforms is not None:
            backup.transforms = BackupTransforms(**transforms)

        train = b.get("train")
        if train is not None:
            bt = BackupTrain(
                tables={k: BackupTrainTable(**v) for k, v in train["tables"].items()}
            )
            backup.train = bt

        generate = b.get("generate")
        if generate is not None:
            backup.generate = BackupGenerate(**generate)

        return backup
