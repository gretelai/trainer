"""Main Trainer Module"""

from __future__ import annotations

import json
import logging
import os.path
from pathlib import Path
from typing import Optional

import pandas as pd
from gretel_client.config import get_session_config, RunnerMode
from gretel_client.projects import create_or_get_unique_project

from gretel_trainer import runner, strategy
from gretel_trainer.models import _BaseConfig, determine_best_model

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")

DEFAULT_PROJECT = "trainer"
DEFAULT_CACHE = f"{DEFAULT_PROJECT}-runner.json"

_ACCEPTABLE_CHARS = set(
    [chr(c) for c in range(ord("a"), ord("z")+1)] +
    [chr(c) for c in range(ord("A"), ord("Z")+1)] +
    [chr(c) for c in range(ord("0"), ord("9")+1)] +
    ["-"]
)
def _sanitize_name(name: str):
    """Replace unacceptable characters for Gretel API project or model names."""
    # Does not account for the following requirements:
    # - must start with [a-zA-Z]
    # - must end with [a-zA-Z0-9]
    # - minimum and maximum length
    return "".join(c if c in _ACCEPTABLE_CHARS else "-" for c in name)

class Trainer:
    """Automated model training and synthetic data generation tool

    Args:
        project_name (str, optional): Gretel project name. Defaults to "trainer".
        model_type (_BaseConfig, optional): Options include GretelLSTM(), GretelACTGAN(). If unspecified, the best option will be chosen at train time based on the training dataset.
        cache_file (str, optional): Select a path to save or load the cache file. Default is `[project_name]-runner.json`.
        overwrite (bool, optional): Overwrite previous progress. Defaults to True.
    """

    def __init__(
        self,
        project_name: str = DEFAULT_PROJECT,
        model_type: Optional[_BaseConfig] = None,
        cache_file: Optional[str] = None,
        overwrite: bool = True,
    ):
        self.dataset_path: Optional[Path] = None
        self.run = None
        self.project_name = project_name
        self.project = create_or_get_unique_project(name=project_name)
        self.overwrite = overwrite
        cache_file = cache_file or f"{project_name}-runner.json"
        self.cache_file = self._get_cache_file(cache_file)
        self.model_type = model_type

        if self.overwrite:
            if self.model_type is None:
                logger.debug("Deferring model configuration to optimize based on training data.")
            else:
                logger.debug(json.dumps(self.model_type.config, indent=2))

    @classmethod
    def load(
        cls, cache_file: str = DEFAULT_CACHE, project_name: str = DEFAULT_PROJECT
    ) -> Trainer:
        """Load an existing project from a cache.

        Args:
            cache_file (str, optional): Valid file path to load the cache file from. Defaults to `[project-name]-runner.json`
            project_name (str, optional): Gretel project name. This should match the original project. Defaults to "trainer"

        Returns:
            Trainer: returns a Trainer instance with an initialized StrategyRunner class.
        """
        if not os.path.exists(cache_file):
            raise ValueError(
                f"Unable to find `{cache_file}`. Please specify a valid cache_file."
            )

        project = create_or_get_unique_project(name=project_name)
        trainer = cls(cache_file=cache_file, project_name=project_name, overwrite=False)

        # Cache file does not store all the data needed to fully rehydrate a StrategyRunner, but
        # 1) we don't need all the attributes to generate data from existing trained models, and
        # 2) if we do want to train from scratch, Trainer#train will create a new StrategyRunner
        #    with legitimate values for these attributes.
        empty_df = pd.DataFrame()
        empty_model_config = {}
        empty_partition_constraints = strategy.PartitionConstraints(max_row_count=0)
        trainer.run = runner.StrategyRunner(
            df=empty_df,
            model_config=empty_model_config,
            partition_constraints=empty_partition_constraints,
            strategy_id=_sanitize_name(trainer._get_strategy_id()),
            cache_file=trainer.cache_file,
            cache_overwrite=trainer.overwrite,
            project=trainer.project,
            hybrid=trainer._hybrid,
        )

        return trainer

    def train(
        self, dataset_path: str, delimiter: str = ",", round_decimals: int = 4, seed_fields: Optional[list] = None,
    ):
        """Train a model on the dataset

        Args:
            dataset_path (str): Path or URL to CSV
            delimiter (str, optional): Delimiter to use when reading the dataset. Defaults to comma (",").
            round_decimals (int, optional): Round decimals in CSV as preprocessing step. Defaults to `4`.
            seed_fields (list, optional): List fields that can be used for conditional generation.
        """
        self.dataset_path = Path(dataset_path)
        self.df = self._preprocess_data(
            dataset_path=dataset_path, delimiter=delimiter, round_decimals=round_decimals
        )
        self.run = self._initialize_run(
            df=self.df, overwrite=self.overwrite, seed_fields=seed_fields
        )
        self.run.train_all_partitions()

    def generate(
        self, num_records: int = 500, seed_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Generate synthetic data

        Args:
            num_records (int, optional): Number of records to generate from model. Defaults to 500.
            seed_df (pd.DataFrame, optional): Pandas DataFrame of values to seed the model with. Defaults to None.

        Returns:
            pd.DataFrame: Synthetic data.
        """
        if self.run is None:
            raise RuntimeError(
                "Cannot generate data without training information. Train a model via `train` or try loading an existing project from cache via `load`."
            )

        self.run.generate_data(
            num_records=num_records if seed_df is None else None,
            max_invalid=None,
            clear_cache=True,
            seed_df=seed_df,
        )
        return self.run.get_synthetic_data()

    def get_sqs_score(self) -> int:
        """Return the average SQS synthetic data quality score.

        Requires the model has been trained.
        """
        if self.run is None:
            raise RuntimeError(
                "Cannot generate data without training information. Train a model via `train` or try loading an existing project from cache via `load`."
            )

        scores = [
            sqs["synthetic_data_quality_score"]["score"]
            for sqs in self.run.get_sqs_information()
        ]
        return int(sum(scores) / len(scores))

    @property
    def _hybrid(self) -> bool:
        return get_session_config().default_runner == RunnerMode.HYBRID

    def _preprocess_data(
        self, dataset_path: str, delimiter: str, round_decimals: int = 4
    ) -> pd.DataFrame:
        """Preprocess input data"""
        tmp = pd.read_csv(dataset_path, sep=delimiter, low_memory=False)
        tmp = tmp.round(round_decimals)
        return tmp

    def _get_cache_file(self, cache_file: str) -> str:
        """Select a path to store the runtime cache to initialize a model"""
        if cache_file is None:
            cache_file = f"{self.project_name}-runner.json"

        if os.path.exists(cache_file):
            if self.overwrite:
                logger.warning(f"Overwriting existing run cache: {cache_file}.")
            else:
                logger.info(f"Using existing run cache: {cache_file}.")
        else:
            logger.info(f"Creating new run cache: {cache_file}.")
        return cache_file

    def _initialize_run(
        self, df: pd.DataFrame, overwrite: bool, seed_fields: Optional[list]
    ) -> runner.StrategyRunner:
        """Create training jobs"""
        if self.model_type is None:
            self.model_type = determine_best_model(df)
            logger.debug(json.dumps(self.model_type.config, indent=2))

        model_config = self.model_type.config

        constraints = strategy.PartitionConstraints(
            max_row_count=self.model_type.max_rows,
            seed_headers=seed_fields,
        )

        strategy_id = self._get_strategy_id()

        run = runner.StrategyRunner(
            strategy_id=_sanitize_name(strategy_id),
            df=self.df,
            cache_file=self.cache_file,
            cache_overwrite=overwrite,
            model_config=model_config,
            partition_constraints=constraints,
            project=self.project,
            hybrid=self._hybrid,
        )
        return run

    def _get_strategy_id(self) -> str:
        strategy_id = self.project_name
        if self.dataset_path is not None:
            strategy_id += f"-{self.dataset_path.stem}"
        return strategy_id
