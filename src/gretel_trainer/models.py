from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import pandas as pd

from gretel_client.projects.models import read_model_config


HIGH_COLUMN_THRESHOLD = 20
HIGH_RECORD_THRESHOLD = 50000
LOW_COLUMN_THRESHOLD = 4
LOW_RECORD_THRESHOLD = 1000


def determine_best_model(df: pd.DataFrame):
    row_count, column_count = df.shape

    if row_count > HIGH_RECORD_THRESHOLD or column_count > HIGH_COLUMN_THRESHOLD:
        return GretelCTGAN(config="synthetics/high-dimensionality")
    elif row_count < LOW_RECORD_THRESHOLD or column_count < LOW_COLUMN_THRESHOLD:
        return GretelCTGAN(config="synthetics/low-record-count")
    else:
        return GretelLSTM(config="synthetics/default")


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
            logging.warning("Privacy filters disabled. Enable with the `enable_privacy_filters` param.")
            self.update_params({"outliers": None, "similarity": None})

        self.validate()

    def update_params(self, params: dict):
        """Convenience function to update model specific parameters from the base config by key value.

        Args:
            params (dict): Dictionary of model parameters and values to update. E.g. {'epochs': 50}
        """
        # Update default config settings with params by key
        for key, value in params.items():
            self.config = self._replace_nested_key(self.config, key, value)
        
    def validate(self):
        if self._model_slug not in list(self.config['models'][0].keys()):
            raise ValueError("Invalid configuration file selected for this model type")

        if self.max_rows > self._max_rows_limit:
            raise ValueError(f"max_rows must be less than {self._max_rows_limit} for this model type.")

        if self.max_header_clusters > self._max_header_clusters_limit:
            raise ValueError(f"max_header_clusters must be less than {self._max_header_clusters_limit} for this model type.")

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