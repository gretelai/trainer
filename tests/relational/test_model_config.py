import pytest

from gretel_client.projects.models import read_model_config
from gretel_trainer.relational.core import MultiTableException
from gretel_trainer.relational.model_config import (
    assemble_configs,
    get_model_key,
    make_evaluate_config,
    make_synthetics_config,
    make_transform_config,
)


def test_get_model_key():
    # Returns the model key when config is valid (at least as far as model key)
    assert get_model_key({"models": [{"amplify": {}}]}) == "amplify"

    # Returns None when given an invalid config
    assert get_model_key({"foo": "bar"}) is None
    assert get_model_key({"models": "wrong type"}) is None
    assert get_model_key({"models": {"wrong": "type"}}) is None
    assert get_model_key({"models": []}) is None
    assert get_model_key({"models": ["wrong type"]}) is None


def test_evaluate_config_prepends_evaluation_type():
    config = make_evaluate_config("users", "individual")
    assert config["name"] == "individual-users"


def test_synthetics_config_prepends_workflow():
    config = make_synthetics_config("users", "synthetics/amplify")
    assert config["name"] == "synthetics-users"


def test_synthetics_config_handles_noncompliant_table_names():
    config = make_synthetics_config("hello--world", "synthetics/amplify")
    assert config["name"] == "synthetics-hello__world"


def test_transform_requires_valid_config(mutagenesis):
    with pytest.raises(MultiTableException):
        make_transform_config(mutagenesis, "atom", "synthetics/amplify")


def test_transform_v2_config_is_unaltered(mutagenesis):
    tv2_config = {
        "schema_version": "1.0",
        "name": "original-name",
        "models": [{"transform_v2": {"some": "Tv2 config"}}],
    }
    config = make_transform_config(mutagenesis, "atom", tv2_config)
    assert config["name"] == "transforms-atom"
    assert config["schema_version"] == tv2_config["schema_version"]
    assert config["models"] == tv2_config["models"]


def test_transforms_config_prepends_workflow(mutagenesis):
    config = make_transform_config(mutagenesis, "atom", "transform/default")
    assert config["name"] == "transforms-atom"


def test_transforms_config_adds_passthrough_policy(mutagenesis):
    def get_policies(config):
        # default blueprint uses `transforms` model key
        return config["models"][0]["transforms"]["policies"]

    original = read_model_config("transform/default")
    original_policies = get_policies(original)

    xform_config = make_transform_config(mutagenesis, "atom", "transform/default")
    xform_config_policies = get_policies(xform_config)

    assert len(xform_config_policies) == len(original_policies) + 1
    assert xform_config_policies[1:] == original_policies
    assert xform_config_policies[0] == {
        "name": "ignore-keys",
        "rules": [
            {
                "name": "ignore-key-columns",
                "conditions": {"field_name": ["atom_id", "molecule_id"]},
                "transforms": [
                    {
                        "type": "passthrough",
                    }
                ],
            }
        ],
    }


_ACTGAN_CONFIG = {"models": [{"actgan": {}}]}
_LSTM_CONFIG = {"models": [{"synthetics": {}}]}
_TABULAR_DP_CONFIG = {"models": [{"tabular_dp": {}}]}


def test_assemble_configs(ecom):
    # Apply a config to all tables
    configs = assemble_configs(
        rel_data=ecom,
        config=_ACTGAN_CONFIG,
        table_specific_configs=None,
        only=None,
        ignore=None,
    )
    assert len(configs) == len(ecom.list_all_tables())
    assert all([config == _ACTGAN_CONFIG for config in configs.values()])

    # Limit scope to a subset of tables
    configs = assemble_configs(
        rel_data=ecom,
        config=_ACTGAN_CONFIG,
        only={"events", "users"},
        table_specific_configs=None,
        ignore=None,
    )
    assert len(configs) == 2

    # Exclude a table
    configs = assemble_configs(
        rel_data=ecom,
        config=_ACTGAN_CONFIG,
        ignore={"events"},
        table_specific_configs=None,
        only=None,
    )
    assert len(configs) == len(ecom.list_all_tables()) - 1

    # Cannot specify both only and ignore
    with pytest.raises(MultiTableException):
        assemble_configs(
            rel_data=ecom,
            config=_ACTGAN_CONFIG,
            only={"users"},
            ignore={"events"},
            table_specific_configs=None,
        )

    # Provide table-specific configs
    configs = assemble_configs(
        rel_data=ecom,
        config=_ACTGAN_CONFIG,
        table_specific_configs={"events": _LSTM_CONFIG},
        only=None,
        ignore=None,
    )
    assert configs["events"] == _LSTM_CONFIG
    assert all(
        [
            config == _ACTGAN_CONFIG
            for table, config in configs.items()
            if table != "events"
        ]
    )

    # Ensure no conflicts between table-specific configs and scope
    with pytest.raises(MultiTableException):
        assemble_configs(
            rel_data=ecom,
            config=_ACTGAN_CONFIG,
            table_specific_configs={"events": _LSTM_CONFIG},
            ignore={"events"},
            only=None,
        )
    with pytest.raises(MultiTableException):
        assemble_configs(
            rel_data=ecom,
            config=_ACTGAN_CONFIG,
            table_specific_configs={"events": _LSTM_CONFIG},
            only={"users"},
            ignore=None,
        )


def test_assemble_configs_json(documents, invented_tables):
    # If table_specific_configs includes a producer table, we apply it to all invented tables
    configs = assemble_configs(
        rel_data=documents,
        config=_ACTGAN_CONFIG,
        table_specific_configs={"purchases": _LSTM_CONFIG},
        only=None,
        ignore=None,
    )
    assert configs == {
        "users": _ACTGAN_CONFIG,
        "payments": _ACTGAN_CONFIG,
        invented_tables["purchases_root"]: _LSTM_CONFIG,
        invented_tables["purchases_data_years"]: _LSTM_CONFIG,
    }

    # If table_specific_configs includes a producer table AND an invented table,
    # the more specific config takes precedence.
    configs = assemble_configs(
        rel_data=documents,
        config=_ACTGAN_CONFIG,
        table_specific_configs={
            "purchases": _LSTM_CONFIG,
            invented_tables["purchases_data_years"]: _TABULAR_DP_CONFIG,
        },
        only=None,
        ignore=None,
    )
    assert configs == {
        "users": _ACTGAN_CONFIG,
        "payments": _ACTGAN_CONFIG,
        invented_tables["purchases_root"]: _LSTM_CONFIG,
        invented_tables["purchases_data_years"]: _TABULAR_DP_CONFIG,
    }

    # Ensure no conflicts between (invented) table-specific configs and scope
    with pytest.raises(MultiTableException):
        assemble_configs(
            rel_data=documents,
            config=_ACTGAN_CONFIG,
            table_specific_configs={
                "purchases": _LSTM_CONFIG,
            },
            ignore={"purchases"},
            only=None,
        )
    with pytest.raises(MultiTableException):
        assemble_configs(
            rel_data=documents,
            config=_ACTGAN_CONFIG,
            table_specific_configs={
                "purchases": _LSTM_CONFIG,
            },
            ignore={invented_tables["purchases_root"]},
            only=None,
        )
    with pytest.raises(MultiTableException):
        assemble_configs(
            rel_data=documents,
            config=_ACTGAN_CONFIG,
            table_specific_configs={
                invented_tables["purchases_root"]: _LSTM_CONFIG,
            },
            ignore={"purchases"},
            only=None,
        )
