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


def test_train_synthetics_custom_config_for_all_tables(ecom, tmpdir, project):
    mock_actgan_config = {"models": [{"actgan": {}}]}

    # We set amplify on the MultiTable instance...
    mt = MultiTable(ecom, project_display_name=tmpdir, gretel_model="amplify")

    # ...but provide an actgan config to train_synthetics.
    mt.train_synthetics(config=mock_actgan_config)

    # The actgan config is used instead of amplify.
    project.create_model_obj.assert_called_with(
        model_config={"name": "synthetics-users", **mock_actgan_config},
        data_source=f"{tmpdir}/synthetics_train_users.csv",
    )


def test_train_synthetics_custom_configs_per_table(ecom, tmpdir, project):
    mock_actgan_config = {"models": [{"actgan": {}}]}
    mock_tabdp_config = {"models": [{"tabular_dp": {}}]}

    # We set amplify on the MultiTable instance...
    mt = MultiTable(ecom, project_display_name=tmpdir, gretel_model="amplify")

    # ...but provide an actgan config to use for tables PLUS a tabular-dp config for one specific table.
    mt.train_synthetics(
        config=mock_actgan_config, table_specific_configs={"events": mock_tabdp_config}
    )

    # The tabular-dp config is used for the singularly called-out table...
    project.create_model_obj.assert_any_call(
        model_config={"name": "synthetics-events", **mock_tabdp_config},
        data_source=f"{tmpdir}/synthetics_train_events.csv",
    )

    # ...and the actgan config is used for all the rest.
    project.create_model_obj.assert_any_call(
        model_config={"name": "synthetics-users", **mock_actgan_config},
        data_source=f"{tmpdir}/synthetics_train_users.csv",
    )


def test_train_synthetics_table_config_and_mt_init_default(ecom, tmpdir, project):
    mock_tabdp_config = {"models": [{"tabular_dp": {}}]}

    # We set amplify on the MultiTable instance...
    mt = MultiTable(ecom, project_display_name=tmpdir, gretel_model="amplify")

    # ...and provide a tabular-dp config for one specific table (but NOT a config).
    mt.train_synthetics(table_specific_configs={"events": mock_tabdp_config})

    # The tabular-dp config is used for the singularly called-out table...
    project.create_model_obj.assert_any_call(
        model_config={"name": "synthetics-events", **mock_tabdp_config},
        data_source=f"{tmpdir}/synthetics_train_events.csv",
    )

    # ...and the amplify blueprint config is used for all the rest.
    project.create_model_obj.assert_any_call(
        model_config=AmplifyConfigMatcher(),
        data_source=f"{tmpdir}/synthetics_train_users.csv",
    )


# Temporary helper to simplify matching an Amplify model config.
# We don't care about recreating the entire Amplify blueprint (which can also change unexpectedly);
# we only care that the model type is Amplify.
# This can be removed once a synthetics config is required by train_synthetics (and we're
# no longer setting or using a gretel_model / blueprint config on the MultiTable instance).
class AmplifyConfigMatcher:
    def __eq__(self, other):
        return list(other["models"][0])[0] == "amplify"


def test_train_synthetics_validates_against_configured_strategy(pets, tmpdir):
    # Independent strategy
    mt_independent = MultiTable(
        pets, project_display_name=tmpdir, strategy="independent"
    )

    mt_independent.train_synthetics(config="synthetics/tabular-lstm")
    mt_independent.train_synthetics(config="synthetics/tabular-actgan")
    mt_independent.train_synthetics(config="synthetics/amplify")
    mt_independent.train_synthetics(config="synthetics/tabular-differential-privacy")
    with pytest.raises(MultiTableException):
        mt_independent.train_synthetics(config="synthetics/time-series")

    # Ancestral strategy
    mt_ancestral = MultiTable(pets, project_display_name=tmpdir, strategy="ancestral")

    mt_ancestral.train_synthetics(config="synthetics/amplify")
    with pytest.raises(MultiTableException):
        mt_ancestral.train_synthetics(config="synthetics/tabular-lstm")
    with pytest.raises(MultiTableException):
        mt_ancestral.train_synthetics(config="synthetics/tabular-actgan")
    with pytest.raises(MultiTableException):
        mt_ancestral.train_synthetics(config="synthetics/tabular-differential-privacy")
    with pytest.raises(MultiTableException):
        mt_ancestral.train_synthetics(config="synthetics/time-series")


def test_train_synthetics_errors(ecom, tmpdir):
    actgan_config = {"models": [{"actgan": {}}]}
    mt = MultiTable(ecom, project_display_name=tmpdir)

    # Invalid config
    with pytest.raises(MultiTableException):
        mt.train_synthetics(config="nonsense")

    # Unrecognized table
    with pytest.raises(MultiTableException):
        mt.train_synthetics(table_specific_configs={"not-a-table": actgan_config})

    # Config provided for omitted table
    with pytest.raises(MultiTableException):
        mt.train_synthetics(
            ignore={"users"}, table_specific_configs={"users": actgan_config}
        )

    # Config for unsupported model
    mt = MultiTable(ecom, project_display_name=tmpdir, strategy="ancestral")
    with pytest.raises(MultiTableException):
        mt.train_synthetics(config=actgan_config)

    # Table config for unsupported model
    mt = MultiTable(ecom, project_display_name=tmpdir, strategy="ancestral")
    with pytest.raises(MultiTableException):
        mt.train_synthetics(table_specific_configs={"users": actgan_config})


def test_train_synthetics_multiple_calls_additive(ecom, tmpdir):
    mt = MultiTable(ecom, project_display_name=tmpdir)
    mt.train_synthetics(only={"products"})
    mt.train_synthetics(only={"users"})

    # We do not lose the first table model
    assert set(mt._synthetics_train.models.keys()) == {"products", "users"}


def test_train_synthetics_models_for_dbs_with_invented_tables(
    documents, tmpdir, get_invented_table_suffix
):
    mt = MultiTable(documents, project_display_name=tmpdir)
    mt.train_synthetics()

    purchases_root_invented_table = f"purchases_{get_invented_table_suffix(1)}"
    purchases_data_years_invented_table = f"purchases_{get_invented_table_suffix(2)}"

    assert set(mt._synthetics_train.models.keys()) == {
        "users",
        "payments",
        purchases_root_invented_table,
        purchases_data_years_invented_table,
    }


def test_train_synthetics_table_filters_cascade_to_invented_tables(documents, tmpdir):
    # When a user provides the ("public") name of a table that contained JSON and led
    # to the creation of invented tables, we recognize that as implicitly applying to
    # all the tables internally created from that source table.
    mt = MultiTable(documents, project_display_name=tmpdir)
    mt.train_synthetics(ignore={"purchases"})

    assert set(mt._synthetics_train.models.keys()) == {"users", "payments"}


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
