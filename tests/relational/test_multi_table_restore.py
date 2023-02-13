import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, Optional
from unittest.mock import Mock, patch

import pytest

import gretel_trainer.relational.backup as b
from gretel_trainer.relational.artifacts import ArtifactCollection
from gretel_trainer.relational.core import MultiTableException, RelationalData
from gretel_trainer.relational.multi_table import MultiTable, TrainStatus

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
    "synthetics_outputs_archive": {
        "artifact_id": "gretel_abc_synthetics_outputs.tar.gz",
        "local_name": "synthetics_outputs.tar.gz",
    },
}


def make_backup(
    rel_data: RelationalData,
    working_dir: Path,
    artifact_collection: ArtifactCollection,
    synthetics_models: Dict[str, Mock] = {},
    synthetics_record_handlers: Dict[str, Mock] = {},
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
    if len(synthetics_models) > 0:
        backup.train = b.BackupTrain(
            tables={
                table: b.BackupTrainTable(
                    model_id=mock.model_id,
                    training_columns=["col1", "col2"],
                )
                for table, mock in synthetics_models.items()
            }
        )
    if len(synthetics_record_handlers) > 0:
        backup.generate = b.BackupGenerate(
            preserved=[],
            record_size_ratio=1.0,
            tables={
                table: b.BackupGenerateTable(record_handler_id=mock.record_handler_id)
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
    synthetics_models: Dict[str, Mock] = {},
    synthetics_record_handlers: Dict[str, Mock] = {},
    output_archive_present: bool = False,
) -> str:
    artifact_collection = ArtifactCollection(
        gretel_debug_summary=ARTIFACTS["debug_summary"]["artifact_id"],
        source_archive=ARTIFACTS["source_archive"]["artifact_id"],
    )
    if output_archive_present:
        artifact_collection.synthetics_outputs_archive = ARTIFACTS[
            "synthetics_outputs_archive"
        ]["artifact_id"]

    backup = make_backup(
        rel_data,
        working_dir,
        artifact_collection,
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


def make_mock_get_model(models: Dict[str, Mock]):
    def get_model(model_id):
        return models[model_id]

    return get_model


def make_mock_model(
    name: str, status: str, record_handler: Optional[Mock] = None
) -> Mock:
    model = Mock()
    model.status = status
    model.model_id = name
    model.data_source = ARTIFACTS[f"train_{name}"]["artifact_id"]
    model.get_report_summary.return_value = {
        "summary": [{"field": "synthetic_data_quality_score", "value": 94}]
    }
    model.get_record_handler.return_value = record_handler
    return model


def make_mock_record_handler(name: str, status: str) -> Mock:
    record_handler = Mock()
    record_handler.status = status
    record_handler.record_handler_id = name
    record_handler.data_source = None
    return record_handler


def local_file(path: Path, identifier: str) -> Path:
    return path / ARTIFACTS[identifier]["local_name"]


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

    # Training CSVs
    for table in rel_data.list_all_tables():
        table_path = setup_path / f"train_{table}.csv"
        rel_data.get_table_data(table).to_csv(table_path, index=False)

    # Synthetics output archive
    with tarfile.open(
        local_file(setup_path, "synthetics_outputs_archive"), "w:gz"
    ) as tar:
        for table in rel_data.list_all_tables():
            table_path = setup_path / f"synth_{table}.csv"
            rel_data.get_table_data(table).to_csv(table_path, index=False)
            tar.add(table_path, arcname=f"synth_{table}.csv")
            for kind, score in [("individual", 90), ("cross_table", 91)]:
                html_filename = f"synthetics_{kind}_evaluation_{table}.html"
                html_path = setup_path / html_filename
                with open(html_path, "w") as f:
                    f.write("<html></html>")
                tar.add(html_path, arcname=html_filename)
                json_filename = f"synthetics_{kind}_evaluation_{table}.json"
                json_path = setup_path / json_filename
                with open(json_path, "w") as f:
                    json.dump(
                        {
                            "summary": [
                                {
                                    "field": "synthetic_data_quality_score",
                                    "value": score,
                                }
                            ]
                        },
                        f,
                    )
                tar.add(json_path, arcname=json_filename)


# For non-archive files, we patch Project#get_artifact_link to return paths to files
# in the test setup dir in place of HTTPS links (smart_open can treat these identically).
# For tar files, though, which require using requests, we patch the entire download_tar_artifact function
def configure_mocks(
    project: Mock,
    download_tar_artifact: Mock,
    setup_path: Path,
    working_path: Path,
    models: Dict[str, Mock] = {},
) -> None:
    project.get_artifact_link = make_mock_get_artifact_link(setup_path)
    project.get_model = make_mock_get_model(models)
    download_tar_artifact.side_effect = make_mock_download_tar_artifact(
        setup_path,
        working_path,
    )


def test_restore_not_yet_trained(project, pets):
    with tempfile.TemporaryDirectory() as working_dir, tempfile.TemporaryDirectory() as testsetup_dir, patch(
        "gretel_trainer.relational.multi_table.download_tar_artifact"
    ) as download_tar_artifact:
        working_dir = Path(working_dir)
        testsetup_dir = Path(testsetup_dir)

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


def test_restore_synthetics_training_still_in_progress(project, pets):
    with tempfile.TemporaryDirectory() as working_dir, tempfile.TemporaryDirectory() as testsetup_dir, patch(
        "gretel_trainer.relational.multi_table.download_tar_artifact"
    ) as download_tar_artifact:
        working_dir = Path(working_dir)
        testsetup_dir = Path(testsetup_dir)

        synthetics_models = {
            "humans": make_mock_model(name="humans", status="active"),
            "pets": make_mock_model(name="pets", status="pending"),
        }

        create_standin_project_artifacts(pets, testsetup_dir)
        configure_mocks(
            project,
            download_tar_artifact,
            testsetup_dir,
            working_dir,
            synthetics_models,
        )
        backup_file = create_backup(pets, working_dir, synthetics_models)

        with pytest.raises(MultiTableException):
            mt = MultiTable.restore(backup_file)


def test_restore_training_complete(project, pets):
    with tempfile.TemporaryDirectory() as working_dir, tempfile.TemporaryDirectory() as testsetup_dir, patch(
        "gretel_trainer.relational.multi_table.download_tar_artifact"
    ) as download_tar_artifact:
        working_dir = Path(working_dir)
        testsetup_dir = Path(testsetup_dir)

        synthetics_models = {
            "humans": make_mock_model(name="humans", status="completed"),
            "pets": make_mock_model(name="pets", status="completed"),
        }

        create_standin_project_artifacts(pets, testsetup_dir)
        configure_mocks(
            project,
            download_tar_artifact,
            testsetup_dir,
            working_dir,
            synthetics_models,
        )
        backup_file = create_backup(pets, working_dir, synthetics_models)

        mt = MultiTable.restore(backup_file)

        # Backup + Debug summary + Source archive + (2) Source CSVs
        # + (2) Train CSVs + (4) Reports
        assert len(os.listdir(working_dir)) == 11

        # Training state is restored
        assert os.path.exists(local_file(working_dir, "train_humans"))
        assert os.path.exists(
            working_dir / "synthetics_individual_evaluation_humans.json"
        )
        assert os.path.exists(
            working_dir / "synthetics_individual_evaluation_humans.html"
        )
        assert os.path.exists(local_file(working_dir, "train_pets"))
        assert os.path.exists(
            working_dir / "synthetics_individual_evaluation_pets.json"
        )
        assert os.path.exists(
            working_dir / "synthetics_individual_evaluation_pets.html"
        )
        assert set(mt.synthetics_train_statuses.values()) == {TrainStatus.Completed}
        assert len(mt._synthetics_models) == 2
        assert mt.evaluations["humans"].individual_sqs == 94
        assert mt.evaluations["humans"].cross_table_sqs is None
        assert mt.evaluations["pets"].individual_sqs == 94
        assert mt.evaluations["pets"].cross_table_sqs is None


def test_restore_training_one_failed(project, pets):
    with tempfile.TemporaryDirectory() as working_dir, tempfile.TemporaryDirectory() as testsetup_dir, patch(
        "gretel_trainer.relational.multi_table.download_tar_artifact"
    ) as download_tar_artifact:
        working_dir = Path(working_dir)
        testsetup_dir = Path(testsetup_dir)

        synthetics_models = {
            "humans": make_mock_model(name="humans", status="error"),
            "pets": make_mock_model(name="pets", status="completed"),
        }

        create_standin_project_artifacts(pets, testsetup_dir)
        configure_mocks(
            project,
            download_tar_artifact,
            testsetup_dir,
            working_dir,
            synthetics_models,
        )
        backup_file = create_backup(pets, working_dir, synthetics_models)

        mt = MultiTable.restore(backup_file)

        # Backup + Debug summary + Source archive + (2) Source CSVs
        # + (1) Train CSVs + (2) Reports
        assert len(os.listdir(working_dir)) == 8

        # Training state is restored
        assert not os.path.exists(local_file(working_dir, "train_humans"))
        assert not os.path.exists(
            working_dir / "synthetics_individual_evaluation_humans.json"
        )
        assert not os.path.exists(
            working_dir / "synthetics_individual_evaluation_humans.html"
        )
        assert os.path.exists(local_file(working_dir, "train_pets"))
        assert os.path.exists(
            working_dir / "synthetics_individual_evaluation_pets.json"
        )
        assert os.path.exists(
            working_dir / "synthetics_individual_evaluation_pets.html"
        )
        assert set(mt.synthetics_train_statuses.values()) == {
            TrainStatus.Completed,
            TrainStatus.Failed,
        }
        assert len(mt._synthetics_models) == 2
        assert mt.evaluations["humans"].individual_sqs is None
        assert mt.evaluations["humans"].cross_table_sqs is None
        assert mt.evaluations["pets"].individual_sqs == 94
        assert mt.evaluations["pets"].cross_table_sqs is None


def test_restore_generate_completed(project, pets):
    with tempfile.TemporaryDirectory() as working_dir, tempfile.TemporaryDirectory() as testsetup_dir, patch(
        "gretel_trainer.relational.multi_table.download_tar_artifact"
    ) as download_tar_artifact:
        working_dir = Path(working_dir)
        testsetup_dir = Path(testsetup_dir)

        synthetics_record_handlers = {
            "humans": make_mock_record_handler(name="humans", status="completed"),
            "pets": make_mock_record_handler(name="pets", status="completed"),
        }

        synthetics_models = {
            "humans": make_mock_model(
                name="humans",
                status="completed",
                record_handler=synthetics_record_handlers["humans"],
            ),
            "pets": make_mock_model(
                name="pets",
                status="completed",
                record_handler=synthetics_record_handlers["pets"],
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
            synthetics_models,
            synthetics_record_handlers,
            output_archive_present=True,
        )

        mt = MultiTable.restore(backup_file)

        # Backup + Debug summary + Source archive + (2) Source CSVs
        # + (2) Train CSVs + (8) Reports + Outputs archive + 2 Synth CSVs
        assert len(os.listdir(working_dir)) == 18

        # Generate state is restored
        assert os.path.exists(working_dir / "synth_humans.csv")
        assert os.path.exists(
            working_dir / "synthetics_cross_table_evaluation_humans.json"
        )
        assert os.path.exists(
            working_dir / "synthetics_cross_table_evaluation_humans.html"
        )
        assert os.path.exists(working_dir / "synth_pets.csv")
        assert os.path.exists(
            working_dir / "synthetics_cross_table_evaluation_pets.json"
        )
        assert os.path.exists(
            working_dir / "synthetics_cross_table_evaluation_pets.html"
        )
        assert set(mt.synthetics_generate_statuses.values()) == {TrainStatus.Completed}
        assert len(mt._synthetics_record_handlers) == 2
        assert len(mt.synthetic_output_tables) == 2
        assert mt.evaluations["humans"].individual_sqs == 90
        assert mt.evaluations["humans"].cross_table_sqs == 91
        assert mt.evaluations["pets"].individual_sqs == 90
        assert mt.evaluations["pets"].cross_table_sqs == 91


def test_restore_generate_in_progress(project, pets):
    with tempfile.TemporaryDirectory() as working_dir, tempfile.TemporaryDirectory() as testsetup_dir, patch(
        "gretel_trainer.relational.multi_table.download_tar_artifact"
    ) as download_tar_artifact:
        working_dir = Path(working_dir)
        testsetup_dir = Path(testsetup_dir)

        synthetics_record_handlers = {
            "humans": make_mock_record_handler(name="humans", status="completed"),
            "pets": make_mock_record_handler(name="pets", status="completed"),
        }

        synthetics_models = {
            "humans": make_mock_model(
                name="humans",
                status="completed",
                record_handler=synthetics_record_handlers["humans"],
            ),
            "pets": make_mock_model(
                name="pets",
                status="completed",
                record_handler=synthetics_record_handlers["pets"],
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
            synthetics_models,
            synthetics_record_handlers,
            output_archive_present=False,
        )

        mt = MultiTable.restore(backup_file)

        # Backup + Debug summary + Source archive + (2) Source CSVs
        # + (2) Train CSVs + (4) Reports
        assert len(os.listdir(working_dir)) == 11

        # Generate state is partially restored
        assert set(mt.synthetics_generate_statuses.values()) == {TrainStatus.Completed}
        assert len(mt._synthetics_record_handlers) == 2
        assert len(mt.synthetic_output_tables) == 0
        assert mt.evaluations["humans"].individual_sqs == 94
        assert mt.evaluations["humans"].cross_table_sqs is None
        assert mt.evaluations["pets"].individual_sqs == 94
        assert mt.evaluations["pets"].cross_table_sqs is None
