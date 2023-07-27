import logging
import time
from typing import Protocol, Union

from gretel_client.projects.jobs import Job, Status
from gretel_client.projects.projects import Project

from gretel_trainer.relational.core import RelationalData
from gretel_trainer.relational.sdk_extras import ExtendedGretelSDK
from gretel_trainer.relational.strategies.ancestral import AncestralStrategy
from gretel_trainer.relational.strategies.independent import IndependentStrategy

logger = logging.getLogger(__name__)


class _MultiTable(Protocol):
    @property
    def _refresh_interval(self) -> int:
        ...

    @property
    def _project(self) -> Project:
        ...

    @property
    def relational_data(self) -> RelationalData:
        ...

    @property
    def _strategy(self) -> Union[AncestralStrategy, IndependentStrategy]:
        ...

    @property
    def _extended_sdk(self) -> ExtendedGretelSDK:
        ...

    def _backup(self) -> None:
        ...


def wait(seconds: int) -> None:
    logger.info(f"Next status check in {seconds} seconds.")
    time.sleep(seconds)


def log_in_progress(table_name: str, status: Status, action: str) -> None:
    logger.info(
        f"{action.capitalize()} job for `{table_name}` still in progress (status: {status})."
    )


def log_success(table_name: str, action: str) -> None:
    logger.info(f"{action.capitalize()} successfully completed for `{table_name}`.")


def log_failed(table_name: str, action: str) -> None:
    logger.info(f"{action.capitalize()} failed for `{table_name}`.")


def log_lost_contact(table_name: str) -> None:
    logger.warning(f"Lost contact with job for `{table_name}`.")


def cleanup(sdk: ExtendedGretelSDK, project: Project, job: Job) -> None:
    sdk.delete_data_source(project, job)
