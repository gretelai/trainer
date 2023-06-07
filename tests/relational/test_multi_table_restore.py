import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import Mock, patch

import pytest

import gretel_trainer.relational.backup as b
from gretel_trainer.relational.artifacts import ArtifactCollection
from gretel_trainer.relational.core import MultiTableException, RelationalData
from gretel_trainer.relational.multi_table import MultiTable, SyntheticsRun

DEBUG_SUMMARY_CONTENT = {"debug": "summary"}

ARTIFACTS = {
    "debug_summary": {
        "artifact_id": "gretel_abc__gretel_debug_summary.json",
        "local_name": "_gretel_debug_summary.json",
    },
    "source_archive": {
        "artifact_id": "gretel_abc_source_tables.tar.gz",
        "local_name": "source_tables.tar.gz",
    },
    "train_humans": {
        "artifact_id": "gretel_abc_train_humans.csv",
        "local_name": "synthetics_train_humans.csv",
    },
    "train_pets": {
        "artifact_id": "gretel_abc_train_pets.csv",
        "local_name": "synthetics_train_pets.csv",
    },
    "synthetics_training_archive": {
        "artifact_id": "gretel_abc_synthetics_training.tar.gz",
        "local_name": "synthetics_training.tar.gz",
    },
    "synthetics_outputs_archive": {
        "artifact_id": "gretel_abc_synthetics_outputs.tar.gz",
        "local_name": "synthetics_outputs.tar.gz",
    },
    "report_humans": {
        "artifact_id": "report",
        "local_name": "synthetics_individual_evaluation_humans.html",
    },
    "report_json_humans": {
        "artifact_id": "report_json",
        "local_name": "synthetics_individual_evaluation_humans.json",
    },
    "report_pets": {
        "artifact_id": "report",
        "local_name": "synthetics_individual_evaluation_pets.html",
    },
    "report_json_pets": {
        "artifact_id": "report_json",
        "local_name": "synthetics_individual_evaluation_pets.json",
    },
}


def make_backup(
    rel_data: RelationalData,
    working_dir: Path,
    artifact_collection: ArtifactCollection,
    transforms_models: dict[str, Mock] = {},
    synthetics_models: dict[str, Mock] = {},
    synthetics_record_handlers: dict[str, Mock] = {},
) -> b.Backup:
    backup = b.Backup(
        project_name="project_name",
        strategy="independent",
        gretel_model="amplify",
        refresh_interval=60,
        working_dir=str(working_dir),
        artifact_collection=artifact_collection,
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
    training_archive_present: bool = False,
    output_archive_present: bool = False,
    transforms_models: dict[str, Mock] = {},
) -> str:
    artifact_collection = ArtifactCollection(
        gretel_debug_summary=ARTIFACTS["debug_summary"]["artifact_id"],
        source_archive=ARTIFACTS["source_archive"]["artifact_id"],
        hybrid=False,
    )
    if training_archive_present:
        artifact_collection.synthetics_training_archive = ARTIFACTS[
            "synthetics_training_archive"
        ]["artifact_id"]
    if output_archive_present:
        artifact_collection.synthetics_outputs_archive = ARTIFACTS[
            "synthetics_outputs_archive"
        ]["artifact_id"]

    backup = make_backup(
        rel_data,
        working_dir,
        artifact_collection,
        transforms_models,
        synthetics_models,
        synthetics_record_handlers,
    )
    return write_backup(backup, working_dir)


def get_local_name(artifact_id):
    local_name = None
    for key, pointers in ARTIFACTS.items():
        if pointers["artifact_id"] == artifact_id:
            local_name = pointers["local_name"]
    if local_name is None:
        raise ValueError
    return local_name


def make_mock_get_artifact_link(setup_path: Path):
    def get_artifact_link(artifact_id):
        return setup_path / get_local_name(artifact_id)

    return get_artifact_link


def make_mock_download_tar_artifact(setup_path: Path, working_path: Path):
    def download_tar_artifact(project, name, out_path):
        local_name = get_local_name(name)

        src = setup_path / local_name
        dest = working_path / local_name
        shutil.copyfile(src, dest)

    return download_tar_artifact


def make_mock_get_model(models: dict[str, Mock]):
    def get_model(model_id):
        return models[model_id]

    return get_model


def make_mock_model(
    name: str,
    status: str,
    setup_path: Path,
    record_handler: Optional[Mock] = None,
    report_json: Optional[dict] = None,
) -> Mock:
    model = Mock()
    model.status = status
    model.model_id = name
    model.data_source = ARTIFACTS[f"train_{name}"]["artifact_id"]
    model.get_artifact_link = make_mock_get_artifact_link(setup_path)
    model.get_record_handler.return_value = record_handler
    return model


def make_mock_record_handler(name: str, status: str) -> Mock:
    record_handler = Mock()
    record_handler.status = status
    record_handler.record_id = name
    record_handler.data_source = None
    return record_handler


def local_file(path: Path, identifier: str) -> Path:
    return path / ARTIFACTS[identifier]["local_name"]


# This is a little gory. Pytest has deprecated importing and calling fixtures directly.
# We want a realistic json blob to use as our backed up report. This copies the blob
# in conftest.py.
_report_json_dict = {
    "synthetic_data_quality_score": {
        "raw_score": 95.86666666666667,
        "grade": "Excellent",
        "score": 95,
    },
    "field_correlation_stability": {
        "raw_score": 0.017275336944403048,
        "grade": "Excellent",
        "score": 94,
    },
    "principal_component_stability": {
        "raw_score": 0.07294532166881013,
        "grade": "Excellent",
        "score": 100,
    },
    "field_distribution_stability": {
        "raw_score": 0.05111941886313614,
        "grade": "Excellent",
        "score": 94,
    },
    "privacy_protection_level": {
        "grade": "Good",
        "raw_score": 2,
        "score": 2,
        "outlier_filter": "Medium",
        "similarity_filter": "Disabled",
        "overfitting_protection": "Enabled",
        "differential_privacy": "Disabled",
    },
    "fatal_error": False,
    "summary": [
        {"field": "synthetic_data_quality_score", "value": 95},
        {"field": "field_correlation_stability", "value": 94},
        {"field": "principal_component_stability", "value": 100},
        {"field": "field_distribution_stability", "value": 94},
        {"field": "privacy_protection_level", "value": 2},
    ],
}


# Create various files in the temporary setup path that stand in for project artifacts in Gretel Cloud
def create_standin_project_artifacts(
    rel_data: RelationalData, setup_path: Path
) -> None:
    # Debug summary
    with open(local_file(setup_path, "debug_summary"), "w") as dbg:
        json.dump(DEBUG_SUMMARY_CONTENT, dbg)

    # Source archive
    with tarfile.open(local_file(setup_path, "source_archive"), "w:gz") as tar:
        for table in rel_data.list_all_tables():
            table_path = setup_path / f"source_{table}.csv"
            rel_data.get_table_data(table).to_csv(table_path, index=False)
            tar.add(table_path, arcname=f"source_{table}.csv")

    # Synthetics training archive
    with tarfile.open(
        local_file(setup_path, "synthetics_training_archive"), "w:gz"
    ) as tar:
        for table in rel_data.list_all_tables():
            table_path = setup_path / f"synthetics_train_{table}.csv"
            rel_data.get_table_data(table).to_csv(table_path, index=False)
            tar.add(table_path, arcname=f"synthetics_train_{table}.csv")

    # Reports
    for table in rel_data.list_all_tables():
        for kind in ["individual", "cross_table"]:
            html_filename = f"synthetics_{kind}_evaluation_{table}.html"
            html_path = setup_path / html_filename
            with open(html_path, "w") as f:
                f.write("<html></html>")
            json_filename = f"synthetics_{kind}_evaluation_{table}.json"
            json_path = setup_path / json_filename
            with open(json_path, "w") as f:
                json.dump(_report_json_dict, f)

    # Synthetics output archive
    # Create a subdirectory with the run outputs
    setup_run_path = setup_path / "run-id"
    os.makedirs(setup_run_path)
    for table in rel_data.list_all_tables():
        table_path = setup_run_path / f"synth_{table}.csv"
        rel_data.get_table_data(table).to_csv(table_path, index=False)
        for kind in ["individual", "cross_table"]:
            html_filename = f"synthetics_{kind}_evaluation_{table}.html"
            json_filename = f"synthetics_{kind}_evaluation_{table}.json"
            shutil.copy(setup_path / html_filename, setup_run_path / html_filename)
            shutil.copy(setup_path / json_filename, setup_run_path / json_filename)

    with tempfile.TemporaryDirectory() as tmpdir:
        runtar = Path(tmpdir) / "run-id"
        # Create the archive for this run...
        shutil.make_archive(
            base_name=str(runtar),
            format="gztar",
            root_dir=setup_run_path,
        )

        # ...and add it to the outputs archive
        with tarfile.open(
            local_file(setup_path, "synthetics_outputs_archive"), "w:gz"
        ) as tar:
            tar.add(f"{runtar}.tar.gz", arcname="run-id.tar.gz")


# For non-archive files, we patch Project#get_artifact_link to return paths to files
# in the test setup dir in place of HTTPS links (smart_open can treat these identically).
# For tar files, though, which require using requests, we patch the entire download_tar_artifact function
def configure_mocks(
    project: Mock,
    download_tar_artifact: Mock,
    setup_path: Path,
    working_path: Path,
    models: dict[str, Mock] = {},
) -> None:
    project.get_artifact_link = make_mock_get_artifact_link(setup_path)
    project.get_model = make_mock_get_model(models)
    download_tar_artifact.side_effect = make_mock_download_tar_artifact(
        setup_path,
        working_path,
    )


@pytest.fixture(autouse=True)
def download_tar_artifact():
    with patch(
        "gretel_trainer.relational.sdk_extras.ExtendedGretelSDK.download_tar_artifact"
    ) as download_tar_artifact:
        yield download_tar_artifact


@pytest.fixture(autouse=True)
def working_dir():
    with tempfile.TemporaryDirectory() as working_dir:
        yield Path(working_dir)


@pytest.fixture(autouse=True)
def testsetup_dir():
    with tempfile.TemporaryDirectory() as testsetup_dir:
        yield Path(testsetup_dir)


def test_restore_not_yet_trained(
    project, pets, download_tar_artifact, working_dir, testsetup_dir
):
    create_standin_project_artifacts(pets, testsetup_dir)
    configure_mocks(project, download_tar_artifact, testsetup_dir, working_dir)
    backup_file = create_backup(pets, working_dir)

    # Restore a MultiTable instance, starting with only the backup file present in working_dir
    assert os.listdir(working_dir) == ["_gretel_backup.json"]
    mt = MultiTable.restore(backup_file)

    # Backup + Debug summary + Source archive + (2) Source CSVs
    assert len(os.listdir(working_dir)) == 5

    # Debug summary is restored
    assert os.path.exists(local_file(working_dir, "debug_summary"))
    with open(local_file(working_dir, "debug_summary"), "r") as dbg:
        assert json.load(dbg) == DEBUG_SUMMARY_CONTENT

    # RelationalData is restored
    assert os.path.exists(working_dir / "source_humans.csv")
    assert os.path.exists(working_dir / "source_pets.csv")
    assert mt.relational_data.debug_summary() == pets.debug_summary()


def test_restore_transforms(
    project, pets, download_tar_artifact, working_dir, testsetup_dir
):
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

    create_standin_project_artifacts(pets, testsetup_dir)
    configure_mocks(
        project,
        download_tar_artifact,
        testsetup_dir,
        working_dir,
        transforms_models,
    )
    backup_file = create_backup(pets, working_dir, transforms_models=transforms_models)

    mt = MultiTable.restore(backup_file)

    # Backup + Debug summary + Source archive + (2) Source CSVs
    assert len(os.listdir(working_dir)) == 5

    # Transforms state is restored
    assert len(mt._transforms_train.models) == 2
    assert len(mt._transforms_train.lost_contact) == 0


def test_restore_synthetics_training_still_in_progress(
    project, pets, download_tar_artifact, working_dir, testsetup_dir
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

    create_standin_project_artifacts(pets, testsetup_dir)
    configure_mocks(
        project,
        download_tar_artifact,
        testsetup_dir,
        working_dir,
        synthetics_models,
    )
    backup_file = create_backup(pets, working_dir, synthetics_models=synthetics_models)

    with pytest.raises(MultiTableException):
        MultiTable.restore(backup_file)


def test_restore_training_complete(
    project, pets, report_json_dict, download_tar_artifact, working_dir, testsetup_dir
):
    synthetics_models = {
        "humans": make_mock_model(
            name="humans",
            status="completed",
            setup_path=testsetup_dir,
            report_json=report_json_dict,
        ),
        "pets": make_mock_model(
            name="pets",
            status="completed",
            setup_path=testsetup_dir,
            report_json=report_json_dict,
        ),
    }

    create_standin_project_artifacts(pets, testsetup_dir)
    configure_mocks(
        project,
        download_tar_artifact,
        testsetup_dir,
        working_dir,
        synthetics_models,
    )
    backup_file = create_backup(
        pets,
        working_dir,
        synthetics_models=synthetics_models,
        training_archive_present=True,
    )

    mt = MultiTable.restore(backup_file)

    # Backup + Debug summary + Source archive + (2) Source CSVs
    # + Training archive + (2) Train CSVs + (4) Reports from models
    assert len(os.listdir(working_dir)) == 12

    # Training state is restored
    assert os.path.exists(local_file(working_dir, "synthetics_training_archive"))
    assert os.path.exists(local_file(working_dir, "train_humans"))
    assert os.path.exists(working_dir / "synthetics_individual_evaluation_humans.json")
    assert os.path.exists(working_dir / "synthetics_individual_evaluation_humans.html")
    assert os.path.exists(local_file(working_dir, "train_pets"))
    assert os.path.exists(working_dir / "synthetics_individual_evaluation_pets.json")
    assert os.path.exists(working_dir / "synthetics_individual_evaluation_pets.html")
    assert len(mt._synthetics_train.models) == 2

    assert mt.evaluations["humans"].individual_sqs == 95
    assert mt.evaluations["humans"].cross_table_sqs is None
    assert mt.evaluations["pets"].individual_sqs == 95
    assert mt.evaluations["pets"].cross_table_sqs is None


def test_restore_training_one_failed(
    project, pets, report_json_dict, download_tar_artifact, working_dir, testsetup_dir
):
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
            report_json=report_json_dict,
        ),
    }

    create_standin_project_artifacts(pets, testsetup_dir)
    configure_mocks(
        project,
        download_tar_artifact,
        testsetup_dir,
        working_dir,
        synthetics_models,
    )
    backup_file = create_backup(
        pets,
        working_dir,
        synthetics_models=synthetics_models,
        training_archive_present=True,
    )

    mt = MultiTable.restore(backup_file)

    # Backup + Debug summary + Source archive + (2) Source CSVs
    # Training archive + (2) Train CSVs + (2) Reports from model
    assert len(os.listdir(working_dir)) == 10

    # Training state is restored
    assert os.path.exists(local_file(working_dir, "synthetics_training_archive"))
    ## We do expect the training CSV to be present, extracted from the training archive...
    assert os.path.exists(local_file(working_dir, "train_humans"))
    ## ...but we should not see evaluation reports because the table failed to train.

    assert not os.path.exists(
        working_dir / "synthetics_individual_evaluation_humans.json"
    )
    assert not os.path.exists(
        working_dir / "synthetics_individual_evaluation_humans.html"
    )

    assert os.path.exists(local_file(working_dir, "train_pets"))
    assert os.path.exists(working_dir / "synthetics_individual_evaluation_pets.json")
    assert os.path.exists(working_dir / "synthetics_individual_evaluation_pets.html")
    assert len(mt._synthetics_train.models) == 2

    assert mt.evaluations["humans"].individual_sqs is None
    assert mt.evaluations["humans"].cross_table_sqs is None
    assert mt.evaluations["pets"].individual_sqs == 95
    assert mt.evaluations["pets"].cross_table_sqs is None


def test_restore_generate_completed(
    project, pets, report_json_dict, download_tar_artifact, working_dir, testsetup_dir
):
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
            report_json=report_json_dict,
        ),
        "pets": make_mock_model(
            name="pets",
            status="completed",
            setup_path=testsetup_dir,
            record_handler=synthetics_record_handlers["pets"],
            report_json=report_json_dict,
        ),
    }

    create_standin_project_artifacts(pets, testsetup_dir)
    configure_mocks(
        project,
        download_tar_artifact,
        testsetup_dir,
        working_dir,
        synthetics_models,
    )
    backup_file = create_backup(
        pets,
        working_dir,
        synthetics_models=synthetics_models,
        synthetics_record_handlers=synthetics_record_handlers,
        training_archive_present=True,
        output_archive_present=True,
    )

    mt = MultiTable.restore(backup_file)

    # Backup + Debug summary + Source archive + (2) Source CSVs
    # + Training archive + (2) Train CSVs + (4) Reports from models
    # + Outputs archive + Previous run subdirectory
    assert len(os.listdir(working_dir)) == 14

    # Generate state is restored
    assert os.path.exists(working_dir / "run-id" / "synth_humans.csv")
    assert os.path.exists(
        working_dir / "run-id" / "synthetics_cross_table_evaluation_humans.json"
    )
    assert os.path.exists(
        working_dir / "run-id" / "synthetics_cross_table_evaluation_humans.html"
    )
    assert os.path.exists(working_dir / "run-id" / "synth_pets.csv")
    assert os.path.exists(
        working_dir / "run-id" / "synthetics_cross_table_evaluation_pets.json"
    )
    assert os.path.exists(
        working_dir / "run-id" / "synthetics_cross_table_evaluation_pets.html"
    )
    assert mt._synthetics_run is not None
    assert len(mt.synthetic_output_tables) == 2
    assert mt.evaluations["humans"].individual_sqs == 95
    assert mt.evaluations["humans"].cross_table_sqs == 95
    assert mt.evaluations["pets"].individual_sqs == 95
    assert mt.evaluations["pets"].cross_table_sqs == 95


def test_restore_generate_in_progress(
    project, pets, report_json_dict, download_tar_artifact, working_dir, testsetup_dir
):
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
            report_json=report_json_dict,
        ),
        "pets": make_mock_model(
            name="pets",
            status="completed",
            setup_path=testsetup_dir,
            record_handler=synthetics_record_handlers["pets"],
            report_json=report_json_dict,
        ),
    }

    create_standin_project_artifacts(pets, testsetup_dir)
    configure_mocks(
        project,
        download_tar_artifact,
        testsetup_dir,
        working_dir,
        synthetics_models,
    )
    backup_file = create_backup(
        pets,
        working_dir,
        synthetics_models=synthetics_models,
        synthetics_record_handlers=synthetics_record_handlers,
        training_archive_present=True,
        output_archive_present=False,
    )

    mt = MultiTable.restore(backup_file)

    # Backup + Debug summary + Source archive + (2) Source CSVs
    # + Training archive + (2) Train CSVs + (4) Reports from models
    assert len(os.listdir(working_dir)) == 12

    # Generate state is partially restored
    assert mt._synthetics_run == SyntheticsRun(
        identifier="run-id",
        preserved=[],
        record_size_ratio=1.0,
        lost_contact=[],
        record_handlers=synthetics_record_handlers,
    )
    assert len(mt.synthetic_output_tables) == 0
    assert mt.evaluations["humans"].individual_sqs == 95
    assert mt.evaluations["humans"].cross_table_sqs is None
    assert mt.evaluations["pets"].individual_sqs == 95
    assert mt.evaluations["pets"].cross_table_sqs is None


def test_restore_hybrid_run(project, pets, report_json_dict, working_dir):
    # In hybrid contexts, the ArtifactCollection does not track or upload anything to the project.
    # We are entirely reliant upon the local directory for those artifacts.
    # For testing, this means we set up everything in the working directory already.

    synthetics_record_handlers = {
        "humans": make_mock_record_handler(name="humans", status="completed"),
        "pets": make_mock_record_handler(name="pets", status="completed"),
    }

    synthetics_models = {
        "humans": make_mock_model(
            name="humans",
            status="completed",
            setup_path=working_dir,
            record_handler=synthetics_record_handlers["humans"],
            report_json=report_json_dict,
        ),
        "pets": make_mock_model(
            name="pets",
            status="completed",
            setup_path=working_dir,
            record_handler=synthetics_record_handlers["pets"],
            report_json=report_json_dict,
        ),
    }

    create_standin_project_artifacts(pets, working_dir)
    download_tar_artifact = Mock()
    configure_mocks(
        project,
        download_tar_artifact,
        working_dir,
        working_dir,
        synthetics_models,
    )

    backup_object = make_backup(
        rel_data=pets,
        working_dir=working_dir,
        artifact_collection=ArtifactCollection(hybrid=True),
        transforms_models={},
        synthetics_models=synthetics_models,
        synthetics_record_handlers=synthetics_record_handlers,
    )
    backup_file = write_backup(backup_object, working_dir)

    mt = MultiTable.restore(backup_file)

    # No need to assert artifacts are present in working_dir because we set it up that way
    # and hybrid restore would not work otherwise.

    assert len(mt._synthetics_train.models) == 2
    assert mt._synthetics_run is not None
    assert len(mt.synthetic_output_tables) == 2
    assert mt.evaluations["humans"].individual_sqs == 95
    assert mt.evaluations["humans"].cross_table_sqs == 95
    assert mt.evaluations["pets"].individual_sqs == 95
    assert mt.evaluations["pets"].cross_table_sqs == 95

    # The ArtifactCollection will not have uploaded any archive files,
    # so restore will not try to download any.
    download_tar_artifact.assert_not_called()
