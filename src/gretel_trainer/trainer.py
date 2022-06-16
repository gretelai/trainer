"""Main Trainer Module"""

import json
import logging
import os.path
from collections import namedtuple
from enum import Enum

import pandas as pd
from gretel_client import ClientConfig, configure_session
from gretel_client.projects import create_or_get_unique_project
from gretel_client.projects.jobs import Status
from gretel_client.projects.models import read_model_config
from gretel_synthetics.utils.header_clusters import cluster

from gretel_trainer import runner, strategy

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")

DEFAULT_PROJECT = "trainer"
DEFAULT_CACHE = f"{DEFAULT_PROJECT}-runner.json"


class ExtendedEnum(Enum):
    """Utility class for Model enum"""

    @classmethod
    def get_types(cls):
        return list(map(lambda c: c.name, cls))

    @classmethod
    def get_config(cls, model: str):
        return cls[model].config


class Model(namedtuple("Model", "config"), ExtendedEnum):
    """Enum to pair valid models and configurations"""

    GretelLSTM = "synthetics/default"
    GretelCTGAN = "synthetics/ctgan"


class Trainer:
    """Automated model training and synthetic data generation tool

    Args:
        project_name (str, optional): Gretel project name. Defaults to "trainer".
        max_header_clusters (int, optional): Max number of clusters per batch. Defaults to 20.
        max_rows (int, optional): Max number of rows per batch. Defaults to 50000.
        model_type (str, optional): Options include ["GretelLSTM", "GretelCTGAN"]. Defaults to "GretelLSTM".
        model_params (dict, optional): Modify model configuration settings by key. E.g. {'epochs': 20}
        cache_file (str, optional): Select a path to save or load the cache file. Default is `[project_name]-runner.json`. 
        overwrite (bool, optional): Overwrite previous progress. Defaults to True.
        enable_privacy_filters (bool, optional): Enable privacy filters on all batches. Warning: On small batches, enabling privacy filters can result in too many records being filtered out at generation time. Defaults to False.
    """

    def __init__(
        self,
        project_name: str = "trainer",
        max_header_clusters: int = 20,
        max_rows: int = 50000,
        model_type: str = "GretelLSTM",
        model_params: dict = {},
        cache_file: str = None,
        overwrite: bool = True,
        enable_privacy_filters: bool = False,
    ):

        configure_session(api_key="prompt", cache="yes", validate=True)

        self.df = None
        self.dataset_path = None
        self.run = None
        self.project_name = project_name
        self.project = create_or_get_unique_project(name=project_name)
        self.max_header_clusters = max_header_clusters
        self.max_rows = max_rows
        self.overwrite = overwrite
        self.cache_file = self._get_cache_file(cache_file)

        if model_type in Model.get_types():
            self.config = read_model_config(Model.get_config(model_type))

            # Update default config settings with params by key
            for key, value in model_params.items():
                self.config = self._replace_nested_key(self.config, key, value)

            if not enable_privacy_filters:
                self.config = self._replace_nested_key(
                    self.config, "outliers", None)
                self.config = self._replace_nested_key(
                    self.config, "similarity", None)

        else:
            raise ValueError(
                f"Invalid model type. Must be {Model.get_model_types()}")

        if self.overwrite:
            logger.debug(json.dumps(self.config, indent=2))

    @classmethod
    def load(
        cls, cache_file: str = DEFAULT_CACHE, project_name: str = DEFAULT_PROJECT
    ) -> runner.StrategyRunner:
        """Load an existing project from a cache.

        Args:
            cache_file (str, optional): Valid file path to load the cache file from. Defaults to `[project-name]-runner.json` 

        Returns:
            Trainer: returns an initialized StrategyRunner class.
        """
        project = create_or_get_unique_project(name=project_name)
        model = cls(cache_file=cache_file,
                    project_name=project_name, overwrite=False)

        if not os.path.exists(cache_file):
            raise ValueError(
                f"Unable to find `{cache_file}`. Please specify a valid cache_file."
            )

        model.run = model._initialize_run(df=None, overwrite=model.overwrite)
        return model

    def train(self, dataset_path: str, round_decimals: int = 4):
        """Train a model on the dataset

        Args:
            dataset_path (str): Path or URL to CSV
            round_decimals (int, optional): Round decimals in CSV as preprocessing step. Defaults to `4`.
        """
        self.dataset_path = dataset_path
        self.df = self._preprocess_data(
            dataset_path=dataset_path, round_decimals=round_decimals
        )
        self.run = self._initialize_run(df=self.df, overwrite=self.overwrite)
        self.run.train_all_partitions()

    def generate(self, num_records: int = 500) -> pd.DataFrame:
        """Generate synthetic data

        Args:
            num_records (int, optional): Number of records to generate from model. Defaults to 500.

        Returns:
            pd.DataFrame: Synthetic data.
        """
        self.run.generate_data(
            num_records=num_records, max_invalid=None, clear_cache=True
        )
        return self.run.get_synthetic_data()

    def _replace_nested_key(self, data, key, value):
        """Replace nested keys"""
        if isinstance(data, dict):
            return {
                k: value if k == key else self._replace_nested_key(
                    v, key, value)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [self._replace_nested_key(v, key, value) for v in data]
        else:
            return data

    def _preprocess_data(
        self, dataset_path: str, round_decimals: int = 4
    ) -> pd.DataFrame:
        """Preprocess input data"""
        tmp = pd.read_csv(dataset_path, low_memory=False)
        tmp = tmp.round(round_decimals)
        return tmp

    def _get_cache_file(self, cache_file: str) -> str:
        """Select a path to store the runtime cache to initialize a model"""
        if cache_file is None:
            cache_file = f"{self.project_name}-runner.json"

        if os.path.exists(cache_file):
            if self.overwrite:
                logger.warning(
                    f"Overwriting existing run cache: {cache_file}.")
            else:
                logger.info(f"Using existing run cache: {cache_file}.")
        else:
            logger.info(f"Creating new run cache: {cache_file}.")
        return cache_file

    def _initialize_run(
        self, df: pd.DataFrame = None, overwrite: bool = True
    ) -> runner.StrategyRunner:
        """Create training jobs"""
        constraints = None
        if df is None:
            df = pd.DataFrame()

        if not df.empty:
            header_clusters = cluster(
                df, maxsize=self.max_header_clusters, plot=False)
            logger.info(
                f"Header clustering created {len(header_clusters)} cluster(s) "
                f"of length(s) {[len(x) for x in header_clusters]}"
            )

            constraints = strategy.PartitionConstraints(
                header_clusters=header_clusters, max_row_count=self.max_rows
            )

        run = runner.StrategyRunner(
            strategy_id=f"{self.project_name}",
            df=self.df,
            cache_file=self.cache_file,
            cache_overwrite=overwrite,
            model_config=self.config,
            partition_constraints=constraints,
            project=self.project,
        )
        return run
