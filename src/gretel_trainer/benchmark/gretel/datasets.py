from __future__ import annotations

import json
from functools import cached_property
from typing import Optional, Union

import boto3
from botocore import UNSIGNED
from botocore.client import Config

from gretel_trainer.benchmark.core import BenchmarkException, Datatype, get_data_shape


class GretelDataset:
    def __init__(self, name: str, datatype: Datatype, tags: list[str]):
        self.name = name
        self.datatype = datatype
        self.tags = tags

    @cached_property
    def data_source(self) -> str:
        return f"https://gretel-datasets.s3.amazonaws.com/{self.name}.csv"

    @cached_property
    def row_count(self) -> int:
        return self._shape[0]

    @cached_property
    def column_count(self) -> int:
        return self._shape[1]

    @cached_property
    def _shape(self) -> tuple[int, int]:
        return get_data_shape(self.data_source)

    def __repr__(self) -> str:
        return f"GretelDataset(name={self.name}, datatype={self.datatype})"

    @property
    def public(self) -> bool:
        return True


class GretelDatasetRepo:
    def __init__(self):
        self.bucket = "gretel-datasets"
        self.s3 = boto3.client(
            "s3", config=Config(signature_version=UNSIGNED), region_name="us-west-2"
        )
        self.manifest = self._read_manifest()

    def list_datasets(
        self,
        datatype: Optional[Union[Datatype, str]] = None,
        tags: Optional[list[str]] = None,
    ) -> list[GretelDataset]:
        matches = list(self.manifest.values())
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
            return self.manifest[name]
        except KeyError:
            raise BenchmarkException(f"No dataset exists with name {name}")

    def list_tags(self) -> list[str]:
        unique_tags = set()
        for dataset in self.manifest.values():
            for tag in dataset.tags:
                unique_tags.add(tag)
        return list(unique_tags)

    def _read_manifest(self) -> dict[str, GretelDataset]:
        response = self.s3.get_object(Bucket=self.bucket, Key="manifest.json")
        data = response["Body"].read()
        manifest = json.loads(data)["datasets"]
        return {
            name: GretelDataset(
                name=name,
                datatype=_coerce_datatype(data["datatype"]),
                tags=data["tags"],
            )
            for name, data in manifest.items()
        }


def _coerce_datatype(datatype: str) -> Datatype:
    if datatype in ("tabular_numeric", "tabular_mixed"):
        return Datatype.TABULAR
    else:
        return Datatype(datatype)
