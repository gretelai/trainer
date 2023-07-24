from __future__ import annotations

import logging
from inspect import isclass
from pathlib import Path
from typing import Optional, Type, Union, cast

import pandas as pd
from gretel_client import configure_session

from gretel_trainer.benchmark.core import BenchmarkConfig, BenchmarkException, Dataset
from gretel_trainer.benchmark.custom.datasets import CustomDataset
from gretel_trainer.benchmark.custom.models import CustomModel
from gretel_trainer.benchmark.gretel.datasets import GretelDataset
from gretel_trainer.benchmark.gretel.models import GretelModel
from gretel_trainer.benchmark.job_spec import JobSpec, model_name
from gretel_trainer.benchmark.session import Session, launch

logger = logging.getLogger(__name__)


DatasetTypes = Union[CustomDataset, GretelDataset]
ModelTypes = Union[
    CustomModel,
    Type[CustomModel],
    GretelModel,
    Type[GretelModel],
]


def compare(
    *,
    datasets: list[DatasetTypes],
    models: list[ModelTypes],
    config: Optional[BenchmarkConfig] = None,
) -> Session:
    config = config or BenchmarkConfig()

    model_instances = _create_models(models)
    _validate_compare(model_instances, datasets)

    # TODO(pm): I think moving this outside of Session makes sense, so it can be configured externally
    configure_session(api_key="prompt", cache="yes", validate=True)

    config.working_dir.mkdir(exist_ok=True)
    standardized_datasets = [
        _standardize_dataset(dataset, config.working_dir) for dataset in datasets
    ]

    jobs = []
    for dataset in standardized_datasets:
        for model in model_instances:
            jobs.append(JobSpec(dataset, model))

    return launch(jobs=jobs, config=config)


def _create_models(models: list[ModelTypes]) -> list[Union[GretelModel, CustomModel]]:
    return [
        cast(Union[GretelModel, CustomModel], m() if isclass(m) else m) for m in models
    ]


def _standardize_dataset(dataset: DatasetTypes, working_dir: Path) -> Dataset:
    source = dataset.data_source
    if isinstance(source, pd.DataFrame):
        csv_path = working_dir / f"{dataset.name}.csv"
        source.to_csv(csv_path, index=False)
        source = str(csv_path)

    return Dataset(
        name=dataset.name,
        datatype=dataset.datatype,
        row_count=dataset.row_count,
        column_count=dataset.column_count,
        data_source=source,
        public=dataset.public,
    )


def _validate_compare(
    all_models: list[Union[GretelModel, CustomModel]], all_datasets: list[DatasetTypes]
) -> None:
    dataset_names = [d.name for d in all_datasets]
    model_names = [model_name(m) for m in all_models]
    _ensure_unique(dataset_names, "datasets")
    _ensure_unique(model_names, "models")


def _ensure_unique(col: list[str], kind: str) -> None:
    if len(set(col)) < len(col):
        raise BenchmarkException(f"{kind} must have unique names")
