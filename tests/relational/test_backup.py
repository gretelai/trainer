import json

from gretel_trainer.relational.artifacts import ArtifactCollection
from gretel_trainer.relational.backup import (
    Backup,
    BackupForeignKey,
    BackupGenerate,
    BackupRelationalData,
    BackupRelationalDataTable,
    BackupSyntheticsTrain,
    BackupTransformsTrain,
)


def test_backup_relational_data(trips):
    expected = BackupRelationalData(
        tables={
            "vehicle_types": BackupRelationalDataTable(primary_key="id"),
            "trips": BackupRelationalDataTable(primary_key="id"),
        },
        foreign_keys=[
            BackupForeignKey(
                foreign_key="trips.vehicle_type_id", referencing="vehicle_types.id"
            )
        ],
    )

    assert BackupRelationalData.from_relational_data(trips) == expected


def test_backup():
    backup_relational = BackupRelationalData(
        tables={
            "customer": BackupRelationalDataTable(
                primary_key="id",
            ),
            "address": BackupRelationalDataTable(
                primary_key=None,
            ),
        },
        foreign_keys=[
            BackupForeignKey(
                foreign_key="address.customer_id", referencing="customer.id"
            )
        ],
    )
    backup_transforms_train = BackupTransformsTrain(
        model_ids={
            "customer": "222333444",
            "address": "888777666",
        },
        lost_contact=[],
    )
    backup_synthetics_train = BackupSyntheticsTrain(
        model_ids={
            "customer": "1234567890",
            "address": "0987654321",
        },
        training_columns={
            "customer": ["id", "first", "last"],
            "address": ["customer_id", "street", "city"],
        },
        lost_contact=[],
    )
    backup_generate = BackupGenerate(
        identifier="run-id",
        preserved=[],
        record_size_ratio=1.0,
        lost_contact=[],
        missing_model=[],
        record_handler_ids={
            "customer": "555444666",
            "address": "333111222",
        },
    )
    artifact_collection = ArtifactCollection(
        gretel_debug_summary="gretel_abc__gretel_debug_summary.json",
        source_archive="gretel_abc_source_tables.tar.gz",
    )
    backup = Backup(
        project_name="my-project",
        strategy="independent",
        gretel_model="amplify",
        working_dir="workdir",
        refresh_interval=120,
        artifact_collection=artifact_collection,
        relational_data=backup_relational,
        transforms_train=backup_transforms_train,
        synthetics_train=backup_synthetics_train,
        generate=backup_generate,
    )

    j = json.dumps(backup.as_dict)
    rehydrated = Backup.from_dict(json.loads(j))

    assert rehydrated == backup
