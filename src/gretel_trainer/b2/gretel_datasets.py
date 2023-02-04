from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import wraps
from typing import Dict, List, Optional, Union

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.client import Config

from gretel_trainer.b2.core import BenchmarkException, Datatype


@dataclass
class GretelDataset:
    name: str
    datatype: Datatype
    tags: List[str]
    row_count: int = field(init=False)

    def __post_init__(self):
        self.row_count = len(pd.read_csv(self.data_source))

    @property
    def data_source(self) -> str:
        return f"https://gretel-datasets.s3.amazonaws.com/{self.name}/data.csv"


class GretelDatasetRepo:
    def __init__(self):
        self.bucket = "gretel-datasets"
        self.s3 = boto3.client(
            "s3", config=Config(signature_version=UNSIGNED), region_name="us-west-2"
        )
        self.datasets = self._read_manifest()

    def list_datasets(
        self,
        datatype: Optional[Union[Datatype, str]] = None,
        tags: Optional[List[str]] = None,
    ) -> List[GretelDataset]:
        matches = list(self.datasets.values())
        if datatype is not None:
            matches = [dataset for dataset in matches if dataset.datatype == datatype]
        if tags is not None:
            matches = [
                dataset
                for dataset in matches
                if all(tag in dataset.tags for tag in tags)
            ]
        return matches

    def get_dataset(self, name: str) -> GretelDataset:
        try:
            return self.datasets[name]
        except KeyError:
            raise BenchmarkException(f"No dataset exists with name {name}")

    def list_tags(self) -> List[str]:
        unique_tags = set()
        for dataset in self.datasets.values():
            for tag in dataset.tags:
                unique_tags.add(tag)
        return list(unique_tags)

    def _read_manifest(self) -> Dict[str, GretelDataset]:
        response = self.s3.get_object(Bucket=self.bucket, Key="manifest.json")
        data = response["Body"].read()
        manifest = json.loads(data)["datasets"]
        return {
            name: GretelDataset(
                name=name,
                datatype=Datatype(data["datatype"]),
                tags=data["tags"],
            )
            for name, data in manifest.items()
        }
