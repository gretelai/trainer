from copy import deepcopy
from typing import Any, Optional

from gretel_client.projects.exceptions import ModelConfigError
from gretel_client.projects.models import read_model_config

from gretel_trainer.relational.core import (
    GretelModelConfig,
    MultiTableException,
    RelationalData,
)


def get_model_key(config_dict: dict[str, Any]) -> Optional[str]:
    try:
        models = config_dict["models"]
        assert isinstance(models, list)
        assert isinstance(models[0], dict)
        return list(models[0])[0]
    except (AssertionError, IndexError, KeyError):
        return None


def ingest(config: GretelModelConfig) -> dict[str, Any]:
    try:
        return read_model_config(deepcopy(config))
    except ModelConfigError as e:
        raise MultiTableException("Invalid config") from e


def _model_name(workflow: str, table: str) -> str:
    ok_table_name = table.replace("--", "__")
    return f"{workflow}-{ok_table_name}"


def make_classify_config(table: str, config: GretelModelConfig) -> dict[str, Any]:
    tailored_config = ingest(config)
    tailored_config["name"] = _model_name("classify", table)
    return tailored_config


def make_evaluate_config(table: str) -> dict[str, Any]:
    tailored_config = ingest("evaluate/default")
    tailored_config["name"] = _model_name("evaluate", table)
    return tailored_config


def make_synthetics_config(table: str, config: GretelModelConfig) -> dict[str, Any]:
    tailored_config = ingest(config)
    tailored_config["name"] = _model_name("synthetics", table)
    return tailored_config


def make_transform_config(
    rel_data: RelationalData, table: str, config: GretelModelConfig
) -> dict[str, Any]:
    tailored_config = ingest(config)
    tailored_config["name"] = _model_name("transforms", table)

    key_columns = rel_data.get_all_key_columns(table)
    if len(key_columns) > 0:
        try:
            model = tailored_config["models"][0]
            try:
                model_key = "transform"
                xform = model[model_key]
            except KeyError:
                model_key = "transforms"
                xform = model[model_key]
            policies = xform["policies"]
        except KeyError:
            raise MultiTableException("Invalid transform config")

        passthrough_policy = _passthrough_policy(key_columns)
        adjusted_policies = [passthrough_policy] + policies

        tailored_config["models"][0][model_key]["policies"] = adjusted_policies

    return tailored_config


def _passthrough_policy(columns: list[str]) -> dict[str, Any]:
    return {
        "name": "ignore-keys",
        "rules": [
            {
                "name": "ignore-key-columns",
                "conditions": {"field_name": columns},
                "transforms": [
                    {
                        "type": "passthrough",
                    }
                ],
            }
        ],
    }
