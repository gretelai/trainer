from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from gretel_trainer.relational.core import ForeignKey


@dataclass
class BackupRelationalDataTable:
    source_artifact_id: str
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


@dataclass
class BackupTrainTable:
    model_id: str
    training_columns: List[str]


@dataclass
class BackupTrain:
    tables: Dict[str, BackupTrainTable]


@dataclass
class BackupGenerateTable:
    record_handler_id: str
    synthetic_artifact_id: Optional[str] = None


@dataclass
class BackupGenerate:
    preserved: List[str]
    record_size_ratio: float
    tables: Dict[str, BackupGenerateTable]


@dataclass
class Backup:
    project_name: str
    strategy: str
    gretel_model: str
    working_dir: str
    refresh_interval: int
    relational_data: BackupRelationalData
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
            relational_data=brd,
        )

        train = b.get("train")
        if train is not None:
            bt = BackupTrain(
                tables={k: BackupTrainTable(**v) for k, v in train["tables"].items()}
            )
            backup.train = bt

        generate = b.get("generate")
        if generate is not None:
            bg = BackupGenerate(
                preserved=generate["preserved"],
                record_size_ratio=generate["record_size_ratio"],
                tables={
                    k: BackupGenerateTable(**v) for k, v in generate["tables"].items()
                },
            )
            backup.generate = bg

        return backup
