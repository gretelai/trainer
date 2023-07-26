from __future__ import annotations

import logging
from inspect import isclass
from pathlib import Path
from typing import Optional, Type, Union, cast

import pandas as pd

from gretel_trainer.benchmark.core import BenchmarkConfig, BenchmarkException, Dataset
from gretel_trainer.benchmark.custom.datasets import CustomDataset
from gretel_trainer.benchmark.custom.models import CustomModel
from gretel_trainer.benchmark.gretel.datasets import GretelDataset
from gretel_trainer.benchmark.gretel.models import GretelModel
from gretel_trainer.benchmark.job_spec import JobSpec, model_name
from gretel_trainer.benchmark.session import Session

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

    config.working_dir.mkdir(exist_ok=True)
    standardized_datasets = [
        _standardize_dataset(dataset, config.working_dir) for dataset in datasets
    ]

    job_specs = []
    for dataset in standardized_datasets:
        for model in models:
            job_specs.append(JobSpec(dataset=dataset, model=_create_model(model)))

    _validate_compare(job_specs)

    session = Session(jobs=job_specs, config=config)
    return session.prepare().execute()


def launch(
    *,
    jobs: list[tuple[DatasetTypes, ModelTypes]],
    config: Optional[BenchmarkConfig] = None,
) -> Session:
    config = config or BenchmarkConfig()

    config.working_dir.mkdir(exist_ok=True)
    job_specs = [
        JobSpec(
            dataset=_standardize_dataset(dataset, config.working_dir),
            model=_create_model(model),
        )
        for dataset, model in jobs
    ]
    session = Session(jobs=job_specs, config=config)
    return session.prepare().execute()


def _create_model(model: ModelTypes) -> Union[GretelModel, CustomModel]:
    return cast(Union[GretelModel, CustomModel], model() if isclass(model) else model)


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


def _validate_compare(jobs: list[JobSpec]) -> None:
    _ensure_unique([job.dataset.name for job in jobs], "datasets")
    _ensure_unique([model_name(job.model) for job in jobs], "models")


def _ensure_unique(col: list[str], kind: str) -> None:
    if len(set(col)) < len(col):
        raise BenchmarkException(f"{kind} must have unique names")