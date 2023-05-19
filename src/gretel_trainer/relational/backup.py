from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

from gretel_trainer.relational.artifacts import ArtifactCollection
from gretel_trainer.relational.core import ForeignKey, RelationalData


@dataclass
class BackupRelationalDataTable:
    primary_key: list[str]
    invented_table_metadata: Optional[dict[str, str]] = None


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
class BackupRelationalJson:
    original_table_name: str
    original_primary_key: list[str]
    original_columns: list[str]
    table_name_mappings: dict[str, str]
    invented_table_names: list[str]


@dataclass
class BackupRelationalData:
    tables: dict[str, BackupRelationalDataTable]
    foreign_keys: list[BackupForeignKey]
    relational_jsons: dict[str, BackupRelationalJson]

    @classmethod
    def from_relational_data(cls, rel_data: RelationalData) -> BackupRelationalData:
        tables = {}
        foreign_keys = []
        relational_jsons = {}
        for table in rel_data.list_all_tables():
            backup_table = BackupRelationalDataTable(
                primary_key=rel_data.get_primary_key(table),
            )
            if (
                invented_table_metadata := rel_data.get_invented_table_metadata(table)
            ) is not None:
                backup_table.invented_table_metadata = asdict(invented_table_metadata)
            tables[table] = backup_table
            foreign_keys.extend(
                [
                    BackupForeignKey.from_fk(key)
                    for key in rel_data.get_foreign_keys(table)
                ]
            )
        for key, rel_json in rel_data.relational_jsons.items():
            relational_jsons[key] = BackupRelationalJson(
                original_table_name=rel_json.original_table_name,
                original_primary_key=rel_json.original_primary_key,
                original_columns=rel_json.original_columns,
                table_name_mappings=rel_json.table_name_mappings,
                invented_table_names=rel_json.table_names,
            )
        return BackupRelationalData(
            tables=tables, foreign_keys=foreign_keys, relational_jsons=relational_jsons
        )


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
            relational_jsons={
                k: BackupRelationalJson(**v)
                for k, v in relational_data.get("relational_jsons", {}).items()
            },
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
