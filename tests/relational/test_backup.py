import json

from gretel_trainer.relational.artifacts import ArtifactCollection
from gretel_trainer.relational.backup import (
    Backup,
    BackupClassify,
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
            "vehicle_types": BackupRelationalDataTable(
                primary_key=["id"],
            ),
            "trips": BackupRelationalDataTable(
                primary_key=["id"],
            ),
        },
        foreign_keys=[
            BackupForeignKey(
                table="trips",
                constrained_columns=["vehicle_type_id"],
                referred_table="vehicle_types",
                referred_columns=["id"],
            )
        ],
    )

    assert BackupRelationalData.from_relational_data(trips) == expected


def test_backup_relational_data_with_json(documents, get_invented_table_suffix):
    purchases_root_invented_table = f"purchases_{get_invented_table_suffix(1)}"
    purchases_data_years_invented_table = f"purchases_{get_invented_table_suffix(2)}"

    expected = BackupRelationalData(
        tables={
            "users": BackupRelationalDataTable(primary_key=["id"]),
            "purchases": BackupRelationalDataTable(
                primary_key=["id"],
                producer_metadata={
                    "invented_root_table_name": purchases_root_invented_table,
                    "table_name_mappings": {
                        "purchases": purchases_root_invented_table,
                        "purchases^data>years": purchases_data_years_invented_table,
                    },
                },
            ),
            purchases_root_invented_table: BackupRelationalDataTable(
                primary_key=["id", "~PRIMARY_KEY_ID~"],
                invented_table_metadata={
                    "invented_root_table_name": purchases_root_invented_table,
                    "original_table_name": "purchases",
                    "json_breadcrumb_path": "purchases",
                    "empty": False,
                },
            ),
            purchases_data_years_invented_table: BackupRelationalDataTable(
                primary_key=["~PRIMARY_KEY_ID~"],
                invented_table_metadata={
                    "invented_root_table_name": purchases_root_invented_table,
                    "original_table_name": "purchases",
                    "json_breadcrumb_path": "purchases^data>years",
                    "empty": False,
                },
            ),
            "payments": BackupRelationalDataTable(primary_key=["id"]),
        },
        foreign_keys=[
            BackupForeignKey(
                table="payments",
                constrained_columns=["purchase_id"],
                referred_table=purchases_root_invented_table,
                referred_columns=["id"],
            ),
            BackupForeignKey(
                table=purchases_root_invented_table,
                constrained_columns=["user_id"],
                referred_table="users",
                referred_columns=["id"],
            ),
            BackupForeignKey(
                table=purchases_data_years_invented_table,
                constrained_columns=["purchases~id"],
                referred_table=purchases_root_invented_table,
                referred_columns=["~PRIMARY_KEY_ID~"],
            ),
        ],
    )

    assert BackupRelationalData.from_relational_data(documents) == expected


def test_backup():
    backup_relational = BackupRelationalData(
        tables={
            "customer": BackupRelationalDataTable(
                primary_key=["id"],
            ),
            "address": BackupRelationalDataTable(
                primary_key=[],
            ),
        },
        foreign_keys=[
            BackupForeignKey(
                table="address",
                constrained_columns=["customer_id"],
                referred_table="customer",
                referred_columns=["id"],
            )
        ],
    )
    backup_classify = BackupClassify(
        model_ids={
            "customer": "aaabbbccc",
            "address": "dddeeefff",
        },
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
        lost_contact=[],
    )
    backup_generate = BackupGenerate(
        identifier="run-id",
        preserved=[],
        record_size_ratio=1.0,
        lost_contact=[],
        record_handler_ids={
            "customer": "555444666",
            "address": "333111222",
        },
    )
    artifact_collection = ArtifactCollection(
        gretel_debug_summary="gretel_abc__gretel_debug_summary.json",
        source_archive="gretel_abc_source_tables.tar.gz",
        hybrid=False,
    )
    backup = Backup(
        project_name="my-project",
        strategy="independent",
        gretel_model="amplify",
        working_dir="workdir",
        refresh_interval=120,
        artifact_collection=artifact_collection,
        relational_data=backup_relational,
        classify=backup_classify,
        transforms_train=backup_transforms_train,
        synthetics_train=backup_synthetics_train,
        generate=backup_generate,
    )

    j = json.dumps(backup.as_dict)
    rehydrated = Backup.from_dict(json.loads(j))

    assert rehydrated == backup
