import json
import os
import shutil
import tempfile

from pathlib import Path
from typing import Optional
from unittest.mock import Mock

import pytest
import smart_open

import gretel_trainer.relational.backup as b

from gretel_trainer.relational.core import MultiTableException, RelationalData
from gretel_trainer.relational.multi_table import MultiTable, SyntheticsRun

SOURCE_ARCHIVE_ARTIFACT_ID = "gretel_abc_source_tables.tar.gz"
SOURCE_ARCHIVE_LOCAL_FILENAME = "source_tables.tar.gz"


def make_backup(
    rel_data: RelationalData,
    source_archive: Optional[str],
    transforms_models: dict[str, Mock] = {},
    synthetics_models: dict[str, Mock] = {},
    synthetics_record_handlers: dict[str, Mock] = {},
) -> b.Backup:
    backup = b.Backup(
        project_name="project_name",
        strategy="independent",
        refresh_interval=60,
        source_archive=source_archive,
        relational_data=b.BackupRelationalData.from_relational_data(rel_data),
    )
    if len(transforms_models) > 0:
        backup.transforms_train = b.BackupTransformsTrain(
            model_ids={
                table: mock.model_id for table, mock in transforms_models.items()
            },
            lost_contact=[],
        )
    if len(synthetics_models) > 0:
        backup.synthetics_train = b.BackupSyntheticsTrain(
            model_ids={
                table: mock.model_id for table, mock in synthetics_models.items()
            },
            lost_contact=[],
        )
    if len(synthetics_record_handlers) > 0:
        backup.generate = b.BackupGenerate(
            identifier="run-id",
            preserved=[],
            record_size_ratio=1.0,
            lost_contact=[],
            record_handler_ids={
                table: mock.record_id
                for table, mock in synthetics_record_handlers.items()
            },
        )
    return backup


def write_backup(backup: b.Backup, out_dir: Path) -> str:
    dest = out_dir / "_gretel_backup.json"
    with open(dest, "w") as b:
        json.dump(backup.as_dict, b)
    return str(dest)


def create_backup(
    rel_data: RelationalData,
    working_dir: Path,
    synthetics_models: dict[str, Mock] = {},
    synthetics_record_handlers: dict[str, Mock] = {},
    transforms_models: dict[str, Mock] = {},
) -> str:
    backup = make_backup(
        rel_data,
        SOURCE_ARCHIVE_ARTIFACT_ID,
        transforms_models,
        synthetics_models,
        synthetics_record_handlers,
    )

    # Clear everything (i.e. original RelationalData source files) from the working directory so that
    # we start the restore process with the dir containing just the backup file and nothing else.
    shutil.rmtree(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)
    return write_backup(backup, working_dir)


def make_mock_get_artifact_handle(setup_path: Path):
    def get_artifact_handle(artifact_id):
        if artifact_id == SOURCE_ARCHIVE_ARTIFACT_ID:
            return smart_open.open(setup_path / SOURCE_ARCHIVE_LOCAL_FILENAME, "rb")
        else:
            raise ValueError(f"unexpected artifact_id: {artifact_id}")

    return get_artifact_handle


def make_mock_get_model(models: dict[str, Mock]):
    def get_model(model_id):
        return models[model_id]

    return get_model


def make_mock_model(
    name: str,
    status: str,
    setup_path: Path,
    record_handler: Optional[Mock] = None,
) -> Mock:
    model = Mock()
    model.status = status
    model.model_id = name
    model.get_artifact_handle = make_mock_get_artifact_handle(setup_path)
    model.get_record_handler.return_value = record_handler
    return model


def make_mock_record_handler(name: str, status: str) -> Mock:
    record_handler = Mock()
    record_handler.status = status
    record_handler.record_id = name
    return record_handler


# Creates a source_archive.tar.gz in the temporary setup path (standing in for Gretel Cloud)
def create_standin_source_archive_artifact(
    rel_data: RelationalData, setup_path: Path
) -> None:
    shutil.make_archive(
        base_name=str(setup_path / SOURCE_ARCHIVE_LOCAL_FILENAME).removesuffix(
            ".tar.gz"
        ),
        format="gztar",
        root_dir=rel_data.source_data_handler.dir,  # type: ignore
    )


def configure_mocks(
    project: Mock,
    setup_path: Path,
    working_path: Path,
    models: dict[str, Mock] = {},
) -> None:
    # The working directory is always named after the project. In these tests we need the name to
    # match the working path that we've configured our other mock artifacts and handlers to use.
    project.name = str(working_path)
    project.get_artifact_handle = make_mock_get_artifact_handle(setup_path)
    project.get_model = make_mock_get_model(models)
    project.artifacts = []


@pytest.fixture(autouse=True)
def working_dir(output_handler):
    return output_handler._working_dir


@pytest.fixture(autouse=True)
def testsetup_dir():
    with tempfile.TemporaryDirectory() as testsetup_dir:
        yield Path(testsetup_dir)


def test_restore_initial_state(project, pets, working_dir, testsetup_dir):
    create_standin_source_archive_artifact(pets, testsetup_dir)
    configure_mocks(project, testsetup_dir, working_dir)
    backup_file = create_backup(pets, working_dir)

    # Restore a MultiTable instance, starting with only the backup file present in working_dir
    assert os.listdir(working_dir) == ["_gretel_backup.json"]
    mt = MultiTable.restore(backup_file)

    # Backup + Source archive + (2) Source CSVs
    assert len(os.listdir(working_dir)) == 4

    # RelationalData is restored
    assert os.path.exists(working_dir / "humans.csv")
    assert os.path.exists(working_dir / "pets.csv")
    assert mt.relational_data.debug_summary() == pets.debug_summary()


def test_restore_transforms(project, pets, working_dir, testsetup_dir):
    transforms_models = {
        "humans": make_mock_model(
            name="humans",
            status="completed",
            setup_path=testsetup_dir,
        ),
        "pets": make_mock_model(
            name="pets",
            status="completed",
            setup_path=testsetup_dir,
        ),
    }

    create_standin_source_archive_artifact(pets, testsetup_dir)
    configure_mocks(
        project,
        testsetup_dir,
        working_dir,
        transforms_models,
    )
    backup_file = create_backup(pets, working_dir, transforms_models=transforms_models)

    mt = MultiTable.restore(backup_file)

    # Transforms state is restored
    assert len(mt._transforms_train.models) == 2
    assert len(mt._transforms_train.lost_contact) == 0


def test_restore_synthetics_training_still_in_progress(
    project, pets, working_dir, testsetup_dir
):
    synthetics_models = {
        "humans": make_mock_model(
            name="humans",
            status="active",
            setup_path=testsetup_dir,
        ),
        "pets": make_mock_model(
            name="pets",
            status="pending",
            setup_path=testsetup_dir,
        ),
    }

    create_standin_source_archive_artifact(pets, testsetup_dir)
    configure_mocks(
        project,
        testsetup_dir,
        working_dir,
        synthetics_models,
    )
    backup_file = create_backup(pets, working_dir, synthetics_models=synthetics_models)

    with pytest.raises(MultiTableException):
        MultiTable.restore(backup_file)


def test_restore_training_complete(project, pets, working_dir, testsetup_dir):
    synthetics_models = {
        "humans": make_mock_model(
            name="humans",
            status="completed",
            setup_path=testsetup_dir,
        ),
        "pets": make_mock_model(
            name="pets",
            status="completed",
            setup_path=testsetup_dir,
        ),
    }

    create_standin_source_archive_artifact(pets, testsetup_dir)
    configure_mocks(
        project,
        testsetup_dir,
        working_dir,
        synthetics_models,
    )
    backup_file = create_backup(
        pets,
        working_dir,
        synthetics_models=synthetics_models,
    )

    mt = MultiTable.restore(backup_file)

    # Training state is restored
    assert len(mt._synthetics_train.models) == 2


def test_restore_training_one_failed(project, pets, working_dir, testsetup_dir):
    synthetics_models = {
        "humans": make_mock_model(
            name="humans",
            status="error",
            setup_path=testsetup_dir,
        ),
        "pets": make_mock_model(
            name="pets",
            status="completed",
            setup_path=testsetup_dir,
        ),
    }

    create_standin_source_archive_artifact(pets, testsetup_dir)
    configure_mocks(
        project,
        testsetup_dir,
        working_dir,
        synthetics_models,
    )
    backup_file = create_backup(
        pets,
        working_dir,
        synthetics_models=synthetics_models,
    )

    mt = MultiTable.restore(backup_file)

    # Training state is restored
    assert len(mt._synthetics_train.models) == 2


def test_restore_generate_completed(project, pets, working_dir, testsetup_dir):
    synthetics_record_handlers = {
        "humans": make_mock_record_handler(name="humans", status="completed"),
        "pets": make_mock_record_handler(name="pets", status="completed"),
    }

    synthetics_models = {
        "humans": make_mock_model(
            name="humans",
            status="completed",
            setup_path=testsetup_dir,
            record_handler=synthetics_record_handlers["humans"],
        ),
        "pets": make_mock_model(
            name="pets",
            status="completed",
            setup_path=testsetup_dir,
            record_handler=synthetics_record_handlers["pets"],
        ),
    }

    create_standin_source_archive_artifact(pets, testsetup_dir)
    configure_mocks(
        project,
        testsetup_dir,
        working_dir,
        synthetics_models,
    )
    backup_file = create_backup(
        pets,
        working_dir,
        synthetics_models=synthetics_models,
        synthetics_record_handlers=synthetics_record_handlers,
    )

    mt = MultiTable.restore(backup_file)

    # Generate task state is restored
    assert mt._synthetics_run == SyntheticsRun(
        identifier="run-id",
        preserved=[],
        record_size_ratio=1.0,
        lost_contact=[],
        record_handlers=synthetics_record_handlers,
    )
    # but note we don't (re)set synthetic_output_tables or evaluations
    assert len(mt.synthetic_output_tables) == 0
    assert mt.evaluations["humans"].individual_sqs is None
    assert mt.evaluations["humans"].cross_table_sqs is None
    assert mt.evaluations["pets"].individual_sqs is None
    assert mt.evaluations["pets"].cross_table_sqs is None
