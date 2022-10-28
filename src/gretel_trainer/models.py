from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import pandas as pd

from gretel_client.projects.models import read_model_config

HIGH_RECORD_THRESHOLD = 1500000
MEDIUM_RECORD_THRESHOLD_2 = 500000
MEDIUM_RECORD_THRESHOLD_1 = 200000
LOW_RECORD_THRESHOLD = 500
HIGH_COLUMN_THRESHOLD = 20
LOW_COLUMN_THRESHOLD = 1


def determine_best_model(manifest: dict):
    """
    [columns]

        | l |                              |                               |
        | o |     high-dimensionality      | high-dimensionality           |  a
        | w |                              |    (low epochs)               |  m
     20 |___|______________________________._______________________________.______
        | r | if text or highly unique:    |                |              |  p
        | e |   "complex-or-free-text"     |                |high-         |  l
        | c | if max_precision > 3:        |                |dimensionality|  i
        | o |   "high-dimensionality"      |                |  (low epochs)|  f
        | r | if 50% columns are numeric:  |                |              |  y
        | d |   "high-dimensionality"      |                |              |
        |   | else:                        |  high-         |              |
        |   |   default                    |  dimensionality|              |
      1 |.__|______________________________.________________.______________.______
        | GPT or Amplify                                                   |
        |___.______________________________.________________.______________.______ [rows]
        0    500                          200k             500k          1.5m

    """

    row_count = manifest["record_count"]
    column_count = manifest["field_count"]
    type_count = {
        type_info["type"]: type_info["count"] for type_info in manifest.get("types", {})
    }
    max_precision = max(
        [field.get("max_precision", 0) for field in manifest.get("fields", {})],
        default=0,
    )
    highly_unique_field_count = sum(
        [
            field.get("unique_percent", 0) > 80
            for field in manifest.get("fields", {})
            if field.get("type") != "numeric"
        ]
    )

    if row_count > HIGH_RECORD_THRESHOLD:
        return GretelAmplify(config="synthetics/amplify", max_rows=row_count)

    elif column_count <= LOW_COLUMN_THRESHOLD:
        if type_count["text"] == LOW_COLUMN_THRESHOLD:
            return GretelGPT(config="synthetics/natural-language", max_rows=row_count)
        else:
            return GretelAmplify(config="synthetics/amplify", max_rows=row_count)

    elif row_count < LOW_RECORD_THRESHOLD:
        return GretelCTGAN("synthetics/low-record-count", max_rows=row_count)

    elif column_count <= HIGH_COLUMN_THRESHOLD:
        if row_count < MEDIUM_RECORD_THRESHOLD_1:
            if type_count.get("other", 0) + type_count.get("text", 0) > 0:
                return GretelLSTM("synthetics/complex-or-free-text")
            elif highly_unique_field_count > 0:
                return GretelLSTM("synthetics/complex-or-free-text")
            elif max_precision > 2:
                return GretelCTGAN("synthetics/high-dimensionality", max_rows=row_count)
            elif type_count.get("numeric", 0) / column_count > 0.5:
                return GretelCTGAN("synthetics/high-dimensionality", max_rows=row_count)
            else:
                return GretelLSTM("synthetics/default")
        elif row_count < MEDIUM_RECORD_THRESHOLD_2:
            return GretelCTGAN("synthetics/high-dimensionality", max_rows=row_count)
        else:
            return GretelCTGAN(
                "https://blueprints-dev.gretel.cloud/config_templates/gretel/synthetics/high-dimensionality-high-record-count.yml",
                max_rows=row_count,
            )
    elif column_count > HIGH_COLUMN_THRESHOLD:
        if row_count < MEDIUM_RECORD_THRESHOLD_1:
            return GretelCTGAN("synthetics/high-dimensionality", max_rows=row_count)
        else:
            return GretelCTGAN(
                "https://blueprints-dev.gretel.cloud/config_templates/gretel/synthetics/high-dimensionality-high-record-count.yml",
                max_rows=row_count,
            )


class _BaseConfig:
    """This class should not be used directly, models should be
    derived from this class
    """

    """This should be overridden on concrete classes"""
    _max_header_clusters_limit: int
    _max_rows_limit: int
    _model_slug: str

    # Should be set by concrete constructors
    config: Union[str, dict]
    max_rows: int
    max_header_clusters: int

    def __init__(
        self,
        config: Union[str, dict],
        max_rows: int,
        max_header_clusters: int,
        enable_privacy_filters: bool,
    ):
        self.config = read_model_config(config)
        self.max_rows = max_rows
        self.max_header_clusters = max_header_clusters
        self.enable_privacy_filters = enable_privacy_filters

        if not self.enable_privacy_filters:
            logging.warning(
                "Privacy filters disabled. Enable with the `enable_privacy_filters` param."
            )
            self.update_params({"outliers": None, "similarity": None})

        self.validate()

    def update_params(self, params: dict):
        """Convenience function to update model specific parameters from the base config by key value.

        Args:
            params (dict): Dictionary of model parameters and values to update. E.g. {"epochs": 50}
        """
        # Update default config settings with params by key
        for key, value in params.items():
            self.config = self._replace_nested_key(self.config, key, value)

    def validate(self):
        if self._model_slug not in list(self.config["models"][0].keys()):
            raise ValueError("Invalid configuration file selected for this model type")

        if self.max_rows > self._max_rows_limit:
            raise ValueError(
                f"max_rows must be less than {self._max_rows_limit} for this model type."
            )

        if self.max_header_clusters > self._max_header_clusters_limit:
            raise ValueError(
                f"max_header_clusters must be less than {self._max_header_clusters_limit} for this model type."
            )

    def _replace_nested_key(self, data, key, value) -> dict:
        """Replace nested keys"""
        if isinstance(data, dict):
            return {
                k: value if k == key else self._replace_nested_key(v, key, value)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [self._replace_nested_key(v, key, value) for v in data]
        else:
            return data


class GretelLSTM(_BaseConfig):
    """
    This model works for a variety of synthetic data tasks including time-series, tabular, and text data. Generally useful for a few thousand records and upward. Dataset generally has a mix of categorical, continuous, and numerical values

    Source data should have <150 columns.

    Args:
        config (str/dict, optional): Either a string representing the path to the config on the local filesystem, a string representing a path to the default Gretel configurations, or a dictionary containing the configurations. Default: "synthetics/default", a default Gretel configuration
        max_rows (int, optional): The number of rows of synthetic data to generate. Defaults to 50000
        max_header_clusters (int, optional): Default: 20
        enable_privacy_filters (bool, optional): Default: False
    """

    _max_header_clusters_limit: int = 30
    _max_rows_limit: int = 5000000
    _model_slug: str = "synthetics"

    def __init__(
        self,
        config="synthetics/default",
        max_rows=50000,
        max_header_clusters=20,
        enable_privacy_filters=False,
    ):
        super().__init__(
            config=config,
            max_rows=max_rows,
            max_header_clusters=max_header_clusters,
            enable_privacy_filters=enable_privacy_filters,
        )


class GretelCTGAN(_BaseConfig):
    """
    This model works well for high dimensional, largely numeric data. Use for datasets with more than 20 columns and/or 50,000 rows.

    Not ideal if dataset contains free text field

    Args:
        config (str/dict, optional): Either a string representing the path to the config on the local filesystem, a string representing a path to the default Gretel configurations, or a dictionary containing the configurations. Default: "synthetics/default", a default Gretel configuration
        max_rows (int, optional): The number of rows of synthetic data to generate. Defaults to 50000
        max_header_clusters (int, optional): Default: 0
        enable_privacy_filters (bool, optional): Default: False
    """

    _max_header_clusters_limit: int = 1
    _max_rows_limit: int = 5000000
    _model_slug: str = "ctgan"

    def __init__(
        self,
        config="synthetics/high-dimensionality",
        max_rows=50000,
        max_header_clusters=0,
        enable_privacy_filters=False,
    ):
        super().__init__(
            config=config,
            max_rows=max_rows,
            max_header_clusters=max_header_clusters,
            enable_privacy_filters=enable_privacy_filters,
        )


class GretelGPT(_BaseConfig):

    _max_header_clusters_limit: int = 1
    _max_rows_limit: int = 5000000
    _model_slug: str = "gpt_x"

    def __init__(
        self,
        config="synthetics/natural-language",
        max_rows=50000,
        max_header_clusters=0,
        enable_privacy_filters=False,
    ):
        super().__init__(
            config=config,
            max_rows=max_rows,
            max_header_clusters=max_header_clusters,
            enable_privacy_filters=enable_privacy_filters,
        )


class GretelAmplify(_BaseConfig):

    _max_header_clusters_limit: int = 1
    _max_rows_limit: int = 5000000
    _model_slug: str = "amplify"

    def __init__(
        self,
        config="synthetics/amplify",
        max_rows=50000,
        max_header_clusters=0,
        enable_privacy_filters=False,
    ):
        super().__init__(
            config=config,
            max_rows=max_rows,
            max_header_clusters=max_header_clusters,
            enable_privacy_filters=enable_privacy_filters,
        )
