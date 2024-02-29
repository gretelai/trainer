from __future__ import annotations

import logging
import shutil

from inspect import isclass
from pathlib import Path
from typing import cast, Optional, Union

import pandas as pd

from gretel_client.config import add_session_context, ClientConfig
from gretel_trainer.benchmark.core import BenchmarkConfig, BenchmarkException, Dataset
from gretel_trainer.benchmark.custom.models import CustomModel
from gretel_trainer.benchmark.gretel.models import GretelModel
from gretel_trainer.benchmark.job_spec import (
    DatasetTypes,
    JobSpec,
    model_name,
    ModelTypes,
)
from gretel_trainer.benchmark.session import Session
from gretel_trainer.version import get_trainer_version

logger = logging.getLogger(__name__)

BENCHMARK_SESSION_METADATA = {"trainer_benchmark": get_trainer_version()}


def compare(
    *,
    datasets: list[DatasetTypes],
    models: list[ModelTypes],
    config: Optional[BenchmarkConfig] = None,
    session: Optional[ClientConfig] = None,
) -> Session:
    cross_product: list[tuple[DatasetTypes, ModelTypes]] = []

    _ensure_unique([d.name for d in datasets], "datasets")
    _ensure_unique([model_name(m) for m in models], "models")

    for dataset in datasets:
        for model in models:
            cross_product.append((dataset, model))

    return _entrypoint(jobs=cross_product, config=config, session=session)


def launch(
    *,
    jobs: list[tuple[DatasetTypes, ModelTypes]],
    config: Optional[BenchmarkConfig] = None,
    session: Optional[ClientConfig] = None,
) -> Session:
    return _entrypoint(jobs=jobs, config=config, session=session)


def _entrypoint(
    *,
    jobs: list[tuple[DatasetTypes, ModelTypes]],
    config: Optional[BenchmarkConfig],
    session: Optional[ClientConfig],
) -> Session:
    session = add_session_context(
        session=session, client_metrics=BENCHMARK_SESSION_METADATA
    )
    _verify_client_config(session)
    config = config or BenchmarkConfig()

    working_dir_cleanup = lambda: None
    if not config.working_dir.exists():
        # clean up working_dir if we create it and we fail to prepare jobs
        working_dir_cleanup = lambda: shutil.rmtree(config.working_dir)
    config.working_dir.mkdir(exist_ok=True)

    try:
        job_specs = [
            JobSpec(
                dataset=_standardize_dataset(dataset, config.working_dir),
                model=_create_model(model),
            )
            for dataset, model in jobs
        ]
    except Exception as e:
        working_dir_cleanup()
        raise e

    session = Session(jobs=job_specs, config=config, session=session)
    return session.prepare().execute()


def _verify_client_config(session: ClientConfig):
    try:
        current_user_email = session.email
        logger.info(f"Using gretel client configured with {current_user_email}!r")
    except Exception as e:
        raise BenchmarkException(
            "Invalid gretel client configuration, please make sure to configure the "
            "client and try again."
        ) from e


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


def _ensure_unique(col: list[str], kind: str) -> None:
    if len(set(col)) < len(col):
        raise BenchmarkException(f"{kind} must have unique names")
