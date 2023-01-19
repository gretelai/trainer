from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Union, Optional

if TYPE_CHECKING:
    import pandas as pd

from gretel_client.projects.models import read_model_config

logger = logging.getLogger(__name__)

HIGH_COLUMN_THRESHOLD = 20
HIGH_RECORD_THRESHOLD = 50_000
LOW_COLUMN_THRESHOLD = 4
LOW_RECORD_THRESHOLD = 1_000


def _actgan_is_best(rows: int, cols: int) -> bool:
    return \
        rows > HIGH_RECORD_THRESHOLD or \
        cols > HIGH_COLUMN_THRESHOLD or \
        rows < LOW_RECORD_THRESHOLD or \
        cols < LOW_COLUMN_THRESHOLD


def determine_best_model(df: pd.DataFrame) -> _BaseConfig:
    """
    Determine the Gretel model best suited for generating synthetic data
    for your dataset.

    Args:
        df (pd.DataFrame): Pandas DataFrame containing the data used to train a synthetic model.

    Returns:
        A Gretel Model object preconfigured for your use case.
    """
    row_count, column_count = df.shape

    if _actgan_is_best(row_count, column_count):
        return GretelACTGAN()
    else:
        return GretelLSTM()


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
    ):
        self.config = read_model_config(config)
        self.max_rows = max_rows
        self.max_header_clusters = max_header_clusters

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
        config (str/dict, optional): Either a string representing the path to the config on the local filesystem, a string representing a path to the default Gretel configurations, or a dictionary containing the configurations. Default: "synthetics/tabular-lstm", a default Gretel configuration
        max_rows (int, optional): The number of rows of synthetic data to generate. Defaults to 50000
        max_header_clusters (int, optional): Default: 20
        enable_privacy_filters (bool, optional): This parameter is deprecated and will be removed in future versions.
    """

    _max_header_clusters_limit: int = 30
    _max_rows_limit: int = 5_000_000
    _model_slug: str = "synthetics"

    def __init__(
        self,
        config="synthetics/tabular-lstm",
        max_rows=50_000,
        max_header_clusters=20,
        enable_privacy_filters=None,
    ):
        _enable_privacy_filters_deprecation_warning(enable_privacy_filters)
        super().__init__(
            config=config,
            max_rows=max_rows,
            max_header_clusters=max_header_clusters,
        )


class GretelACTGAN(_BaseConfig):
    """
    This model works well for high dimensional, largely numeric data. Use for datasets with more than 20 columns and/or 50,000 rows.

    Not ideal if dataset contains free text field

    Args:
        config (str/dict, optional): Either a string representing the path to the config on the local filesystem, a string representing a path to the default Gretel configurations, or a dictionary containing the configurations. Default: "synthetics/tabular-actgan", a default Gretel configuration
        max_rows (int, optional): The number of rows of synthetic data to generate. Defaults to 50000
        max_header_clusters (int, optional): Default: 500
        enable_privacy_filters (bool, optional): This parameter is deprecated and will be removed in future versions.
    """

    _max_header_clusters_limit: int = 5_000
    _max_rows_limit: int = 5_000_000
    _model_slug: str = "actgan"

    def __init__(
        self,
        config="synthetics/tabular-actgan",
        max_rows=1_000_000,
        max_header_clusters=1_000,
        enable_privacy_filters=None,
    ):
        _enable_privacy_filters_deprecation_warning(enable_privacy_filters)
        super().__init__(
            config=config,
            max_rows=max_rows,
            max_header_clusters=max_header_clusters,
        )


def _enable_privacy_filters_deprecation_warning(value: Optional[bool]) -> None:
    if value is not None:
        logger.warning(
            "Privacy filters are now configured as `auto` and `enable_privacy_filters` "
            "parameter is deprecated and will be removed in future versions. "
            "See https://docs.gretel.ai/gretel.ai/synthetics/synthetics-faqs/privacy-protection#primary-protection-filters for more information"
        )


class GretelAmplify(_BaseConfig):
    """
    This model is able to generate large quantities of data from real-world data or synthetic data.

    Note: this model doesn't currently support privacy filtering.

    Args:
        config (str/dict, optional): Either a string representing the path to the config on the local filesystem, a string representing a path to the default Gretel configurations, or a dictionary containing the configurations. Default: "synthetics/amplify", a default Gretel configuration for Amplify.
        max_rows (int, optional): The number of rows of synthetic data to generate. Defaults to 50000
        max_header_clusters (int, optional): Default: 50
    """

    _max_header_clusters_limit: int = 1_000
    _max_rows_limit: int = 1_000_000_000
    _model_slug: str = "amplify"

    def __init__(
        self,
        config="synthetics/amplify",
        max_rows=50_000,
        max_header_clusters=500,
    ):
        super().__init__(
            config=config,
            max_rows=max_rows,
            max_header_clusters=max_header_clusters,
        )
