from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Optional, Type, Union

import gretel_trainer.benchmark.compare as c
import gretel_trainer.benchmark.custom.datasets
import pandas as pd

from gretel_trainer.benchmark.core import Dataset, Datatype, ModelFactory
from gretel_trainer.benchmark.gretel.datasets import GretelDataset, GretelPublicDatasetRepo
from gretel_trainer.benchmark.gretel.evaluate import GretelEvaluate
from gretel_trainer.benchmark.gretel.models import (
    GretelAmplify,
    GretelAuto,
    GretelCTGAN,
    GretelGPTX,
    GretelLSTM,
    GretelModel,
)
from gretel_trainer.benchmark.gretel.sdk import ActualGretelSDK
from gretel_trainer import trainer

BENCHMARK_DIR = "./.benchmark"

repo = GretelPublicDatasetRepo(
    bucket="gretel-datasets",
    region="us-west-2",
    load_dir=f"{BENCHMARK_DIR}/gretel_datasets",
)


def get_gretel_dataset(name: str) -> GretelDataset:
    return repo.get_dataset(name)


def list_gretel_datasets(
    datatype: Optional[Union[Datatype, str]] = None, tags: Optional[List[str]] = None
) -> List[GretelDataset]:
    return repo.list_datasets(datatype, tags)


def list_gretel_dataset_tags() -> List[str]:
    return repo.list_tags()


def make_dataset(
    sources: Union[List[str], List[pd.DataFrame]],
    *,
    datatype: Union[Datatype, str],
    namespace: Optional[str] = None,
    delimiter: str = ",",
) -> Dataset:
    return gretel_trainer.benchmark.custom.datasets.make_dataset(
        sources,
        datatype=datatype,
        namespace=namespace,
        delimiter=delimiter,
        local_dir=BENCHMARK_DIR,
    )


def compare(
    *,
    datasets: List[Dataset],
    models: List[Union[ModelFactory, Type[GretelModel]]],
) -> c.Comparison:
    return c.compare(
        datasets=datasets,
        models=models,
        runtime_config=c.RuntimeConfig(
            cleanup_wait_secs=10,
            local_dir=BENCHMARK_DIR,
            project_prefix=_timestamp,
            thread_pool=ThreadPoolExecutor(4),
        ),
        gretel_sdk=ActualGretelSDK,
        evaluator=GretelEvaluate(),
        gretel_trainer_factory=trainer.Trainer,
    )


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")
