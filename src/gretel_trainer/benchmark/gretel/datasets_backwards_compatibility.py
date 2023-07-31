# This module exclusively contains deprecated functions that only exist for backwards compatibility
# It can be deleted completely once we fully remove these functions.

import logging
from typing import Optional, Union

from gretel_trainer.benchmark import Datatype
from gretel_trainer.benchmark.gretel.datasets import GretelDataset, GretelDatasetRepo

logger = logging.getLogger(__name__)


def get_gretel_dataset(name: str) -> GretelDataset:
    _deprecation_warning("get_gretel_dataset", "get_dataset")
    repo = GretelDatasetRepo()
    return repo.get_dataset(name)


def list_gretel_datasets(
    datatype: Optional[Union[Datatype, str]] = None, tags: Optional[list[str]] = None
) -> list[GretelDataset]:
    _deprecation_warning("list_gretel_datasets", "list_datasets")
    repo = GretelDatasetRepo()
    return repo.list_datasets(datatype, tags)


def list_gretel_dataset_tags() -> list[str]:
    _deprecation_warning("list_gretel_dataset_tags", "list_tags")
    repo = GretelDatasetRepo()
    return repo.list_tags()


def _deprecation_warning(old: str, new: str) -> None:
    logger.warning(
        f"`{old}` as a freestanding function is deprecated, and will be removed in a future release. "
        f"To avoid this warning message, going forward you should create a `GretelDatasetRepo` instance and call the `{new}` method."
    )
