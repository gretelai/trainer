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

HIGH_COLUMN_THRESHOLD = 2
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
      1 |.__|______________________________.________________._____________.______
        | GPT or Amplify                   |                |             |
        |___.______________________________.________________._____________.______ [rows]
        0    500                          200k             500k          1.5m

    """

    row_count = manifest["record_count"]
    column_count = manifest["field_count"]
    type_count = {
        manifest["types"][k]["type"]: manifest["types"][k]["count"]
        for k in range(len(manifest["types"]))
    }
    max_precision = max(
        [x["max_precision"] for x in manifest["fields"] if "max_precision" in x.keys()]
        + [0]
    )
    highly_unique_field_count = manifest["highly_unique_field_count"]

    if column_count <= LOW_COLUMN_THRESHOLD:
        if type_count["text"] == LOW_COLUMN_THRESHOLD:
            return GretelGPT(config="synthetics/natural-language")
        else:
            return GretelAmplify(config="synthetics/amplify")

    elif row_count > HIGH_RECORD_THRESHOLD:
        return GretelAmplify(config="synthetics/amplify")

    elif row_count < LOW_RECORD_THRESHOLD:
        return GretelCTGAN("synthetics/low-record-count")

    elif column_count <= HIGH_COLUMN_THRESHOLD:
        if row_count < MEDIUM_RECORD_THRESHOLD_1:
            if type_count["other"] + type_count["text"] > 0:
                return GretelLSTM("synthetics/complex-or-free-text")
            elif highly_unique_field_count > 0:
                return GretelLSTM("synthetics/complex-or-free-text")
            elif max_precision > 2:
                return GretelCTGAN("synthetics/high-dimensionality")
            elif type_count["numeric"] / column_count > 0.5:
                return GretelCTGAN("synthetics/high-dimensionality")
            else:
                return GretelLSTM("synthetics/default")
        elif row_count < MEDIUM_RECORD_THRESHOLD_2:
            ## change this based on either n_rows * n_cols or n_rows/n_cols
            return GretelCTGAN("synthetics/high-dimensionality")
        else:
            ## change this based on either n_rows * n_cols or n_rows/n_cols
            return GretelCTGAN("synthetics/high-dimensionality-high-records")
    elif column_count > HIGH_COLUMN_THRESHOLD:
        if row_count < MEDIUM_RECORD_THRESHOLD_1:
            return GretelCTGAN("synthetics/high-dimensionality")
        else:
            ## change to high-dim-high-records when available
            return GretelCTGAN("synthetics/high-dimensionality-high-records")


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

    _max_header_clusters_limit: int = 1000
    _max_rows_limit: int = 5000000
    _model_slug: str = "ctgan"

    def __init__(
        self,
        config="synthetics/high-dimensionality",
        max_rows=50000,
        max_header_clusters=500,
        enable_privacy_filters=False,
    ):
        super().__init__(
            config=config,
            max_rows=max_rows,
            max_header_clusters=max_header_clusters,
            enable_privacy_filters=enable_privacy_filters,
        )


class GretelGPT(_BaseConfig):

    _max_header_clusters_limit: int = 1000
    _max_rows_limit: int = 5000000
    _model_slug: str = "gpt_x"

    def __init__(
        self,
        config="synthetics/natural-language",
        enable_privacy_filters=False,
    ):
        super().__init__(
            config=config,
            enable_privacy_filters=enable_privacy_filters,
        )


class GretelAmplify(_BaseConfig):

    _max_header_clusters_limit: int = 1000
    _max_rows_limit: int = 5000000
    _model_slug: str = "amplify"

    def __init__(
        self,
        config="synthetics/amplify",
        enable_privacy_filters=False,
    ):
        super().__init__(
            config=config,
            enable_privacy_filters=enable_privacy_filters,
        )
