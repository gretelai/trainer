import tarfile
import tempfile
from pathlib import Path
from unittest.mock import Mock

from gretel_trainer.relational.artifacts import ArtifactCollection, add_to_tar


def test_makes_new_archive():
    with tempfile.TemporaryDirectory() as tmpdir, tempfile.NamedTemporaryFile() as tf1, tempfile.NamedTemporaryFile() as tf2:
        archive_path = Path(tmpdir) / "archive.tar.gz"
        add_to_tar(archive_path, Path(tf1.name), "tf1")

        with tarfile.open(archive_path, "r:gz") as tar:
            assert len(tar.getnames()) == 1


def test_appends_to_existing_archive():
    with tempfile.TemporaryDirectory() as tmpdir, tempfile.NamedTemporaryFile() as tf1, tempfile.NamedTemporaryFile() as tf2:
        archive_path = Path(tmpdir) / "archive.tar.gz"
        add_to_tar(archive_path, Path(tf1.name), "tf1")
        add_to_tar(archive_path, Path(tf2.name), "tf2")

        with tarfile.open(archive_path, "r:gz") as tar:
            assert len(tar.getnames()) == 2


def test_uploads_path_to_project_and_stores_artifact_key():
    ac = ArtifactCollection(hybrid=False)
    project = Mock()
    project.upload_artifact.return_value = "artifact_key"

    with tempfile.NamedTemporaryFile() as tmpfile:
        ac.upload_gretel_debug_summary(project=project, path=tmpfile.name)

    project.upload_artifact.assert_called_once_with(tmpfile.name)
    assert ac.gretel_debug_summary == "artifact_key"


def test_overwrites_project_artifacts():
    ac = ArtifactCollection(hybrid=False, gretel_debug_summary="first_key")
    project = Mock()
    project.upload_artifact.return_value = "second_key"

    with tempfile.NamedTemporaryFile() as tmpfile:
        ac.upload_gretel_debug_summary(project=project, path=tmpfile.name)

    project.upload_artifact.assert_called_once_with(tmpfile.name)
    project.delete_artifact.assert_called_once_with("first_key")
    assert ac.gretel_debug_summary == "second_key"


def test_does_not_upload_in_hybrid_mode():
    ac = ArtifactCollection(hybrid=True)
    project = Mock()

    with tempfile.NamedTemporaryFile() as tmpfile:
        ac.upload_gretel_debug_summary(project=project, path=tmpfile.name)

    project.upload_artifact.assert_not_called()
    assert ac.gretel_debug_summary is None
