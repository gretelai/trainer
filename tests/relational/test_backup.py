import json

from gretel_trainer.relational.backup import (
    Backup,
    BackupForeignKey,
    BackupGenerate,
    BackupGenerateTable,
    BackupRelationalData,
    BackupRelationalDataTable,
    BackupTrain,
    BackupTrainTable,
)


def test_backup():
    backup_relational = BackupRelationalData(
        tables={
            "customer": BackupRelationalDataTable(
                source_artifact_id="gretel_abcdefg_customer.csv",
                primary_key="id",
            ),
            "address": BackupRelationalDataTable(
                source_artifact_id="gretel_abcdefg_address.csv",
                primary_key=None,
            ),
        },
        foreign_keys=[
            BackupForeignKey(
                foreign_key="address.customer_id", referencing="customer.id"
            )
        ],
    )
    backup_train = BackupTrain(
        tables={
            "customer": BackupTrainTable(
                model_id="1234567890",
                training_columns=["id", "first", "last"],
            ),
            "address": BackupTrainTable(
                model_id="0987654321",
                training_columns=["customer_id", "street", "city"],
            ),
        }
    )
    backup_generate = BackupGenerate(
        preserved=[],
        record_size_ratio=1.0,
        tables={
            "customer": BackupGenerateTable(
                record_handler_id="555444666",
                synthetic_artifact_id="gretel_boaief_synth_customer.csv",
            ),
            "address": BackupGenerateTable(
                record_handler_id="333111222",
                synthetic_artifact_id=None,
            ),
        },
    )
    backup = Backup(
        project_name="my-project",
        strategy="independent",
        gretel_model="amplify",
        working_dir="workdir",
        refresh_interval=120,
        relational_data=backup_relational,
        train=backup_train,
        generate=backup_generate,
    )

    j = json.dumps(backup.as_dict)
    rehydrated = Backup.from_dict(json.loads(j))

    assert rehydrated == backup
