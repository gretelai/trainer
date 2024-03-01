import logging
import time

from gretel_client.projects.jobs import Job, Status
from gretel_client.projects.projects import Project
from gretel_trainer.relational.sdk_extras import ExtendedGretelSDK

logger = logging.getLogger(__name__)


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
    sdk.delete_data_sources(project, job)
