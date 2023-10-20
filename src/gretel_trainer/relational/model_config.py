from copy import deepcopy
from typing import Any, Optional

from gretel_client.projects.exceptions import ModelConfigError
from gretel_client.projects.models import read_model_config
from gretel_trainer.relational.core import (
    GretelModelConfig,
    MultiTableException,
    RelationalData,
)

TRANSFORM_MODEL_KEYS = ["transform", "transforms", "transform_v2"]


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


def make_evaluate_config(table: str, sqs_type: str) -> dict[str, Any]:
    tailored_config = ingest("evaluate/default")
    tailored_config["name"] = _model_name(sqs_type, table)
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

    model_key, model = next(iter(tailored_config["models"][0].items()))

    # Ensure we have a transform config
    if model_key not in TRANSFORM_MODEL_KEYS:
        raise MultiTableException("Invalid transform config")

    # Tv2 configs pass through unaltered (except for name, above)
    if model_key == "transform_v2":
        return tailored_config

    # We add a passthrough policy to Tv1 configs to avoid transforming PK/FK columns
    key_columns = rel_data.get_all_key_columns(table)
    if len(key_columns) > 0:
        policies = model["policies"]
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


def assemble_configs(
    rel_data: RelationalData,
    config: GretelModelConfig,
    table_specific_configs: Optional[dict[str, GretelModelConfig]],
    only: Optional[set[str]],
    ignore: Optional[set[str]],
) -> dict[str, Any]:
    only, ignore = _expand_only_and_ignore(rel_data, only, ignore)

    tables_in_scope = [
        table
        for table in rel_data.list_all_tables()
        if not _skip_table(table, only, ignore)
    ]

    # Standardize type of all provided models
    config_dict = ingest(config)
    table_specific_config_dicts = {
        table: ingest(conf) for table, conf in (table_specific_configs or {}).items()
    }

    # Translate any JSON-source tables in table_specific_configs to invented tables
    all_table_specific_config_dicts = {}
    for table, conf in table_specific_config_dicts.items():
        m_names = rel_data.get_modelable_table_names(table)
        if len(m_names) == 0:
            raise MultiTableException(f"Unrecognized table name: `{table}`")
        for m_name in m_names:
            all_table_specific_config_dicts[m_name] = table_specific_config_dicts.get(
                m_name, conf
            )

    # Ensure compatibility between only/ignore and table_specific_configs
    omitted_tables_with_overrides_specified = []
    for table in all_table_specific_config_dicts:
        if _skip_table(table, only, ignore):
            omitted_tables_with_overrides_specified.append(table)
    if len(omitted_tables_with_overrides_specified) > 0:
        raise MultiTableException(
            f"Cannot provide configs for tables that have been omitted from synthetics training: "
            f"{omitted_tables_with_overrides_specified}"
        )

    return {
        table: all_table_specific_config_dicts.get(table, config_dict)
        for table in tables_in_scope
    }


def _expand_only_and_ignore(
    rel_data: RelationalData, only: Optional[set[str]], ignore: Optional[set[str]]
) -> tuple[Optional[set[str]], Optional[set[str]]]:
    """
    Accepts the `only` and `ignore` parameter values as provided by the user and:
    - ensures both are not set (must provide one or the other, or neither)
    - translates any JSON-source tables to the invented tables
    """
    if only is not None and ignore is not None:
        raise MultiTableException("Cannot specify both `only` and `ignore`.")

    modelable_tables = set()
    for table in only or ignore or {}:
        m_names = rel_data.get_modelable_table_names(table)
        if len(m_names) == 0:
            raise MultiTableException(f"Unrecognized table name: `{table}`")
        modelable_tables.update(m_names)

    if only is None:
        return (None, modelable_tables)
    elif ignore is None:
        return (modelable_tables, None)
    else:
        return (None, None)


def _skip_table(
    table: str, only: Optional[set[str]], ignore: Optional[set[str]]
) -> bool:
    skip = False
    if only is not None and table not in only:
        skip = True
    if ignore is not None and table in ignore:
        skip = True

    return skip
