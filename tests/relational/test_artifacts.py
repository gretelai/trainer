import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from gretel_trainer.relational.artifacts import (
    ArtifactCollection,
    archive_items,
    archive_nested_dir,
)


@pytest.fixture()
def out_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture()
def tempfiles():
    with tempfile.NamedTemporaryFile() as tf1, tempfile.NamedTemporaryFile() as tf2:
        yield [Path(tf1.name), Path(tf2.name)]


def test_archiving_items(out_dir, tempfiles):
    tgz = out_dir / "out.tar.gz"
    archive_items(tgz, tempfiles)
    print(tgz.exists())

    with tarfile.open(tgz, "r:gz") as tar:
        assert set(tar.getnames()) == {
            ".",
            f"./{tempfiles[0].name}",
            f"./{tempfiles[1].name}",
        }


def test_archive_nested_dir(out_dir, tempfiles):
    tgz = out_dir / "out.tar.gz"

    with tempfile.TemporaryDirectory() as tmpdir:
        dir_to_archive = Path(tmpdir)
        for tfile in tempfiles:
            shutil.copy(tfile, dir_to_archive)

        # When the provided targz path does not yet exist, creates a new archive with a nested archive
        archive_nested_dir(tgz, dir_to_archive, "id1")

        with tarfile.open(tgz, "r:gz") as tar:
            assert set(tar.getnames()) == {".", "./id1.tar.gz"}

            with tempfile.TemporaryDirectory() as td:
                tar.extract("./id1.tar.gz", td)
                with tarfile.open(f"{td}/id1.tar.gz", "r:gz") as nested:
                    assert set(nested.getnames()) == {
                        ".",
                        f"./{tempfiles[0].name}",
                        f"./{tempfiles[1].name}",
                    }

        # Later, new directories can be added to the existing archive
        archive_nested_dir(tgz, dir_to_archive, "id2")

        with tarfile.open(tgz, "r:gz") as tar:
            assert set(tar.getnames()) == {".", "./id1.tar.gz", "./id2.tar.gz"}

        # In the case of a name conflict, the added dir overwrites/replaces an existing dir
        archive_nested_dir(tgz, dir_to_archive, "id2")

        with tarfile.open(tgz, "r:gz") as tar:
            assert set(tar.getnames()) == {".", "./id1.tar.gz", "./id2.tar.gz"}


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
