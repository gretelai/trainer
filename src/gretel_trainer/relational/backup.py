from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional, Union

from gretel_trainer.relational.artifacts import ArtifactCollection
from gretel_trainer.relational.core import ForeignKey, RelationalData, Scope
from gretel_trainer.relational.json import InventedTableMetadata, ProducerMetadata


@dataclass
class BackupRelationalDataTable:
    primary_key: list[str]
    invented_table_metadata: Optional[dict[str, Any]] = None
    producer_metadata: Optional[dict[str, Any]] = None


@dataclass
class BackupForeignKey:
    table: str
    constrained_columns: list[str]
    referred_table: str
    referred_columns: list[str]

    @classmethod
    def from_fk(cls, fk: ForeignKey) -> BackupForeignKey:
        return BackupForeignKey(
            table=fk.table_name,
            constrained_columns=fk.columns,
            referred_table=fk.parent_table_name,
            referred_columns=fk.parent_columns,
        )


@dataclass
class BackupRelationalData:
    tables: dict[str, BackupRelationalDataTable]
    foreign_keys: list[BackupForeignKey]

    @classmethod
    def from_relational_data(cls, rel_data: RelationalData) -> BackupRelationalData:
        tables = {}
        foreign_keys = []
        for table in rel_data.list_all_tables(Scope.ALL):
            tables[table] = BackupRelationalDataTable(
                primary_key=rel_data.get_primary_key(table),
                invented_table_metadata=_optionally_as_dict(
                    rel_data.get_invented_table_metadata(table)
                ),
                producer_metadata=_optionally_as_dict(
                    rel_data.get_producer_metadata(table)
                ),
            )

            # Producer tables delegate their foreign keys to root invented tables.
            # We exclude producers here to avoid adding duplicate foreign keys.
            if not rel_data.is_producer_of_invented_tables(table):
                foreign_keys.extend(
                    [
                        BackupForeignKey.from_fk(key)
                        for key in rel_data.get_foreign_keys(table)
                    ]
                )
        return BackupRelationalData(tables=tables, foreign_keys=foreign_keys)


def _optionally_as_dict(
    metadata: Optional[Union[InventedTableMetadata, ProducerMetadata]]
) -> Optional[dict[str, Any]]:
    if metadata is None:
        return None

    return asdict(metadata)


@dataclass
class BackupClassify:
    model_ids: dict[str, str]


@dataclass
class BackupTransformsTrain:
    model_ids: dict[str, str]
    lost_contact: list[str]


@dataclass
class BackupSyntheticsTrain:
    model_ids: dict[str, str]
    lost_contact: list[str]


@dataclass
class BackupGenerate:
    identifier: str
    preserved: list[str]
    record_size_ratio: float
    record_handler_ids: dict[str, str]
    lost_contact: list[str]


@dataclass
class Backup:
    project_name: str
    strategy: str
    gretel_model: str
    working_dir: str
    refresh_interval: int
    artifact_collection: ArtifactCollection
    relational_data: BackupRelationalData
    classify: Optional[BackupClassify] = None
    transforms_train: Optional[BackupTransformsTrain] = None
    synthetics_train: Optional[BackupSyntheticsTrain] = None
    generate: Optional[BackupGenerate] = None

    @property
    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, b: dict[str, Any]):
        relational_data = b["relational_data"]
        brd = BackupRelationalData(
            tables={
                k: BackupRelationalDataTable(**v)
                for k, v in relational_data.get("tables", {}).items()
            },
            foreign_keys=[
                BackupForeignKey(
                    table=fk["table"],
                    constrained_columns=fk["constrained_columns"],
                    referred_table=fk["referred_table"],
                    referred_columns=fk["referred_columns"],
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

        classify = b.get("classify")
        if classify is not None:
            backup.classify = BackupClassify(**classify)

        transforms_train = b.get("transforms_train")
        if transforms_train is not None:
            backup.transforms_train = BackupTransformsTrain(**transforms_train)

        synthetics_train = b.get("synthetics_train")
        if synthetics_train is not None:
            backup.synthetics_train = BackupSyntheticsTrain(**synthetics_train)

        generate = b.get("generate")
        if generate is not None:
            backup.generate = BackupGenerate(**generate)

        return backup
