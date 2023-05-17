import tempfile
from unittest.mock import ANY, patch

import pytest

from gretel_trainer.relational.multi_table import MultiTable


# The assertions in this file are concerned with setting up the transforms train
# workflow state properly, and stop short of kicking off the task.
@pytest.fixture(autouse=True)
def run_task():
    with patch("gretel_trainer.relational.multi_table.run_task"):
        yield


@pytest.fixture(autouse=True)
def backup():
    with patch.object(MultiTable, "_backup", return_value=None):
        yield


@pytest.fixture()
def tmpdir(project):
    with tempfile.TemporaryDirectory() as tmpdir:
        project.name = tmpdir
        yield tmpdir


def test_train_transforms_defaults_to_transforming_all_tables(ecom, tmpdir):
    mt = MultiTable(ecom, project_display_name=tmpdir)
    mt.train_transforms("transform/default")
    transforms_train = mt._transforms_train

    assert set(transforms_train.models.keys()) == set(ecom.list_all_tables())


def test_train_transforms_only_includes_specified_tables(ecom, tmpdir, project):
    mt = MultiTable(ecom, project_display_name=tmpdir)
    mt.train_transforms("transform/default", only=["users"])
    transforms_train = mt._transforms_train

    assert set(transforms_train.models.keys()) == {"users"}
    project.create_model_obj.assert_called_with(
        model_config=ANY,  # a tailored transforms config, in dict form
        data_source=f"{tmpdir}/transforms_train_users.csv",
    )


def test_train_transforms_ignore_excludes_specified_tables(ecom, tmpdir):
    mt = MultiTable(ecom, project_display_name=tmpdir)
    mt.train_transforms("transform/default", ignore=["distribution_center", "products"])
    transforms_train = mt._transforms_train

    assert set(transforms_train.models.keys()) == {
        "events",
        "users",
        "order_items",
        "inventory_items",
    }


# The public method under test here is deprecated
def test_train_transform_models(ecom, tmpdir):
    mt = MultiTable(ecom, project_display_name=tmpdir)
    mt.train_transform_models(
        configs={
            "events": "transform/default",
            "users": "transform/default",
        }
    )
    transforms_train = mt._transforms_train

    assert set(transforms_train.models.keys()) == {"events", "users"}
