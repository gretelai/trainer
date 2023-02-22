import tarfile
import tempfile
from pathlib import Path

from gretel_trainer.relational.artifacts import add_to_tar


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
