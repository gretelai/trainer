import tempfile
from unittest.mock import ANY, patch

import pytest

from gretel_trainer.relational.core import MultiTableException
from gretel_trainer.relational.multi_table import MultiTable


# The assertions in this file are concerned with setting up the synthetics train
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


def test_train_synthetics_defaults_to_training_all_tables(ecom, tmpdir):
    mt = MultiTable(ecom, project_display_name=tmpdir)
    mt.train_synthetics()

    assert set(mt._synthetics_train.models.keys()) == set(ecom.list_all_tables())


def test_train_synthetics_only_includes_specified_tables(ecom, tmpdir, project):
    mt = MultiTable(ecom, project_display_name=tmpdir)
    mt.train_synthetics(only={"users"})

    assert set(mt._synthetics_train.models.keys()) == {"users"}
    project.create_model_obj.assert_called_with(
        model_config=ANY,  # a tailored synthetics config, in dict form
        data_source=f"{tmpdir}/synthetics_train_users.csv",
    )


def test_train_synthetics_ignore_excludes_specified_tables(ecom, tmpdir):
    mt = MultiTable(ecom, project_display_name=tmpdir)
    mt.train_synthetics(ignore={"distribution_center", "products"})

    assert set(mt._synthetics_train.models.keys()) == {
        "events",
        "users",
        "order_items",
        "inventory_items",
    }


def test_train_synthetics_exits_early_if_unrecognized_tables(ecom, tmpdir, project):
    mt = MultiTable(ecom, project_display_name=tmpdir)
    with pytest.raises(MultiTableException):
        mt.train_synthetics(ignore={"nonsense"})

    assert len(mt._synthetics_train.models) == 0
    project.create_model_obj.assert_not_called()


def test_train_synthetics_multiple_calls_additive(ecom, tmpdir):
    mt = MultiTable(ecom, project_display_name=tmpdir)
    mt.train_synthetics(only={"products"})
    mt.train_synthetics(only={"users"})

    # We do not lose the first table model
    assert set(mt._synthetics_train.models.keys()) == {"products", "users"}


def test_train_synthetics_multiple_calls_overwrite(ecom, tmpdir, project):
    project.create_model_obj.return_value = "m1"

    mt = MultiTable(ecom, project_display_name=tmpdir)
    mt.train_synthetics(only={"products"})

    assert mt._synthetics_train.models["products"] == "m1"

    project.reset_mock()
    project.create_model_obj.return_value = "m2"

    # calling a second time will create a new model for the table that overwrites the original
    mt.train_synthetics(only={"products"})
    assert mt._synthetics_train.models["products"] == "m2"


def test_train_synthetics_after_deleting_models(ecom, tmpdir):
    mt = MultiTable(ecom, project_display_name=tmpdir)
    mt.train_synthetics(only={"products"})
    mt.delete_models("synthetics")
    mt.train_synthetics(only={"users"})

    assert set(mt._synthetics_train.models.keys()) == {"users"}

    # You can scope deletion to a subset of tables
    mt.train_synthetics(only={"users", "products"})
    assert set(mt._synthetics_train.models.keys()) == {"users", "products"}
    mt.delete_models("synthetics", ignore={"products"})
    assert set(mt._synthetics_train.models.keys()) == {"products"}
