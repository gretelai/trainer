"""Main Trainer Module"""

from collections import namedtuple
from enum import Enum
import json
import logging
import pandas as pd
import pkgutil
import yaml

from gretel_client import configure_session, ClientConfig
from gretel_client.projects import create_or_get_unique_project
from gretel_client.projects.models import read_model_config
from gretel_client.projects.jobs import Status
from gretel_synthetics.utils.header_clusters import cluster

from gretel_trainer import strategy, runner


logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


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

    GretelLSTM = "templates/gretel_lstm.yaml"
    GretelCTGAN = "templates/gretel_ctgan.yaml"


class Trainer:
    """Automated model training and synthetic data generation tool

    Args:
        project_name (str, optional): Gretel project name. Defaults to "trainer".
        max_header_clusters (int, optional): Max number of clusters per batch. Defaults to 20.
        max_rows (int, optional): Max number of rows per batch. Defaults to 50000.
        model_type (str, optional): Options include ["GretelLSTM", "GretelCTGAN"]. Defaults to "GretelLSTM".
        model_params (dict, optional): Modify model configuration settings by key. E.g. {'epochs': 20}
    """

    def __init__(
        self,
        project_name: str = "trainer",
        max_header_clusters: int = 20,
        max_rows: int = 50000,
        model_type: str = "GretelLSTM",
        model_params: dict = {}
    ):

        configure_session(api_key="prompt", cache="yes", validate=True)

        self.df = None
        self.dataset_path = None
        self.project_name = project_name
        self.project = create_or_get_unique_project(name=project_name)
        self.max_header_clusters = max_header_clusters
        self.max_rows = max_rows

        if model_type in Model.get_types():
            config = pkgutil.get_data(__name__, Model.get_config(model_type))
            self.config = yaml.load(config, Loader=yaml.FullLoader)

            # Update default config settings with kwargs by key
            for key, value in model_params.items():
                self.config = self.replace_nested_key(self.config, key, value)
            logger.debug(json.dumps(self.config,indent=2))

        else:
            raise ValueError(
                f"Invalid model type. Must be {Model.get_model_types()}")

    def train(self, dataset_path: str, overwrite: bool = True, round_decimals: int = 4):
        """Train a model on the dataset

        Args:
            dataset_path (str): Path or URL to CSV
            overwrite (bool, optional): Overwrite previous progress. Defaults to True.
            round_decimals (int, optional): Round decimals in CSV as preprocessing step. Defaults to 4.
        """
        self.dataset_path = dataset_path
        self.df = self._preprocess_data(
            dataset_path=dataset_path, round_decimals=round_decimals
        )
        self.run = self._initialize_run(df=self.df, overwrite=overwrite)
        self.run.train_all_partitions()

    def load(self):
        """Load an existing strategy"""
        self.run = self._initialize_run(overwrite=False)

    def generate(self, num_records: int = 500) -> pd.DataFrame:
        """Generate synthetic data"""
        self.run.generate_data(num_records=num_records, max_invalid=None)
        return self.run.get_synthetic_data()
 
    def replace_nested_key(self, data, key, value):
        """Replace nested keys"""
        if isinstance(data, dict):
            return {
                k: value if k == key else self.replace_nested_key(v, key, value)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [self.replace_nested_key(v, key, value) for v in data]
        else:
            return data

    def _preprocess_data(
        self, dataset_path: str, round_decimals: int = 4
    ) -> pd.DataFrame:
        """Preprocess input data"""
        tmp = pd.read_csv(dataset_path, low_memory=False)
        tmp = tmp.round(round_decimals)
        return tmp

    def _initialize_run(
        self, df: pd.DataFrame = pd.DataFrame(), overwrite: bool = True
    ) -> runner.StrategyRunner:
        """Create training jobs"""
        constraints = None

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
            cache_file=f"{self.project_name}-runner.json",
            cache_overwrite=overwrite,
            model_config=self.config,
            partition_constraints=constraints,
            project=self.project,
        )
        return run
