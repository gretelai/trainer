from __future__ import annotations

import csv
import json
import os

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from typing_extensions import Protocol

import boto3

from botocore import UNSIGNED
from botocore.client import BaseClient, Config
from gretel_trainer.benchmark.core import BenchmarkException, Datatype

GRETEL_DELIMITER = ","


@dataclass
class GretelCSVSource:
    name: str
    path: str
    datatype: Datatype
    row_count: int
    column_count: int

    @property
    def delimiter(self) -> str:
        return GRETEL_DELIMITER

    @contextmanager
    def unwrap(self):
        yield self.path


class GretelDataset:
    def __init__(
        self,
        name: str,
        datatype: Datatype,
        tags: List[str],
        s3: BaseClient,
        bucket: str,
        load_dir: str,
    ):
        self.name = name
        self.datatype = datatype
        self.tags = tags
        self.s3 = s3
        self.bucket = bucket
        self.load_dir = load_dir

    def sources(self) -> List[GretelCSVSource]:
        _sources = self._load()
        if len(_sources) == 0:
            raise BenchmarkException(f"Could not load sources for dataset {self.name}")
        return _sources

    def _load(self) -> List[GretelCSVSource]:
        prefix = f"{self.name}/"
        list_s3_objects = self.s3.list_objects_v2(
            Bucket=self.bucket,
            Prefix=prefix,
        )
        keys = [
            obj["Key"] for obj in list_s3_objects["Contents"] if obj["Key"] != prefix
        ]
        os.makedirs(f"{self.load_dir}/{self.name}", exist_ok=True)
        sources = []
        for key in keys:
            source_name = key
            local_path = f"{self.load_dir}/{key}"
            self.s3.download_file(Bucket=self.bucket, Key=key, Filename=local_path)
            shape = _get_data_shape(local_path)
            sources.append(
                GretelCSVSource(
                    name=source_name,
                    path=local_path,
                    datatype=self.datatype,
                    row_count=shape[0],
                    column_count=shape[1],
                )
            )
        return sources


def _get_data_shape(path: str) -> Tuple[int, int]:
    with open(os.path.expanduser(path)) as f:
        reader = csv.reader(f, delimiter=GRETEL_DELIMITER)
        cols = len(next(reader))
        # We just read the header to get cols,
        # so the remaining rows are all data
        rows = sum(1 for _ in f)

    return (rows, cols)


class GretelDatasetRepo(Protocol):
    def list_datasets(
        self, datatype: Optional[Union[Datatype, str]], tags: Optional[List[str]]
    ) -> List[GretelDataset]:
        ...

    def get_dataset(self, name: str) -> GretelDataset:
        ...

    def list_tags(self) -> List[str]:
        ...


class GretelPublicDatasetRepo:
    def __init__(self, bucket: str, region: str, load_dir: str):
        self.bucket = bucket
        self.s3 = boto3.client(
            "s3", config=Config(signature_version=UNSIGNED), region_name=region
        )
        self.load_dir = load_dir
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
                s3=self.s3,
                bucket=self.bucket,
                load_dir=self.load_dir,
            )
            for name, data in manifest.items()
        }
